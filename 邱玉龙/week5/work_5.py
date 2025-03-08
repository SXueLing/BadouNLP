#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances,cosine_similarity
#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # print(vector)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        # print(vector)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    cluster_centers = kmeans.cluster_centers_
    distances = cosine_distances(vectors, cluster_centers)  # 使用余弦距离计算所有点到质心的距离矩阵

    # 计算每个类的平均距离
    labers = kmeans.labels_
    from collections import defaultdict
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, labers):  #取出句子和标签
        sentence_label_dict[label].append(sentence)

    intra_class_distances = []
    for cluster_id in range(n_clusters):
        cluster_distances = distances[labers == cluster_id, cluster_id]  # 获取属于该类的样本到质心的距离
        avg_distance = np.mean(cluster_distances) if len(cluster_distances) > 0 else 0  # 平均值
        intra_class_distances.append((cluster_id, avg_distance))
    intra_class_distances.sort(key=lambda x: x[1])
    for cluster_id, avg_distance in intra_class_distances:
        print(f"类{cluster_id}的平均距离为：{avg_distance:.4f}")
        print(sentence_label_dict[cluster_id])
        print("==========================")


    # 方式二 公式计算
    # cluster_avg_distances = []
    # for cluster_id in range(n_clusters):
    #     # 获取属于当前聚类的样本
    #     cluster_samples = vectors[labers == cluster_id]
    #     # print(cluster_samples[0])
    #     centroid = cluster_centers[cluster_id]
    #     # print(centroid)
    #     cluster_distances = []
    #     for sample in cluster_samples:
    #         value = 1 - cosine_similarity(sample.reshape(1, -1), centroid.reshape(1, -1))[0][0]
    #         cluster_distances.append(value)
    #     avg_distance = np.mean(cluster_distances) if len(cluster_distances) > 0 else 0
    #     cluster_avg_distances.append((cluster_id, avg_distance))
    # cluster_avg_distances.sort(key=lambda x: x[1])
    # 
    # for cluster_id, avg_distance in cluster_avg_distances:
    #     print(f"类{cluster_id}的平均距离为：{avg_distance:.4f}")
    #     print(sentence_label_dict[cluster_id])
    #     print("==========================")



if __name__ == "__main__":
    main()

