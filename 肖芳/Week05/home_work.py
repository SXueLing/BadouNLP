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
from collections import defaultdict

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
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    sentence_list = np.array(list(sentences))
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    # print('vectors len:', len(vectors))

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算


    # print('kmeans.labels_ == 0:', kmeans.labels_ == 0)
    # print('vectors cluster 0:', vectors[kmeans.labels_ == 0])
    # print('sentences cluster 0:', sentence_list[kmeans.labels_ == 0])
    # 将聚类结果按照类内平均距离排序
    cluster_distances = []
    for i in range(n_clusters):
        # print('kmeans.labels_:',kmeans.labels_ == i)
        # 获取当前类别的所有样本
        cluster_vectors = vectors[kmeans.labels_ == i]
        # 计算到聚类中心的距离
        distances = np.linalg.norm(cluster_vectors - kmeans.cluster_centers_[i], axis=1)
        # 计算平均距离
        avg_distance = np.mean(distances)
        cluster_distances.append((i, avg_distance))
    
    # 按照平均距离排序    
    sorted_clusters = sorted(cluster_distances, key=lambda x: x[1])

    for cluster_idx, avg_dist in sorted_clusters:
        print(f"\nCluster {cluster_idx} (Average distance: {avg_dist:.4f}):")
        cluster_sentences = [s for s, l in zip(sentences, kmeans.labels_) if l == cluster_idx]
        for i in range(min(10, len(cluster_sentences))):
            print(cluster_sentences[i])
        print("---------")

if __name__ == "__main__":
    main()

