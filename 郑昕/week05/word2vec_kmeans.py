#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
# import re
# import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

from sklearn.metrics import euclidean_distances

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
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    # === 计算各个样本到所有聚类中心的距禃 ===
    # distances 的形状是 （样本数，n_clusters）, 其中元素表示样本到各个聚类中心的欧式距离
    distances = euclidean_distances(vectors, kmeans.cluster_centers_)

    # 下面统计每个类内样本的距离之和，再除以该类样本数，即得到类内平均距离
    cluster_distance_sum = [0.0] * n_clusters
    cluster_count = [0] * n_clusters

    for i, label in enumerate(kmeans.labels_):
        # i 代表样本索引, label 即该样本所属的聚类
        # distances[i][label] 即样本 i 到其所属类中心的距离
        cluster_distance_sum[label] += distances[i][label]
        cluster_count[label] += 1

    # 打印每个类的平均距离
    cluster_avg_distance = []
    for label in range(n_clusters):
        if cluster_count[label] == 0:
            avg_distance = 0
        else:
            avg_distance = cluster_distance_sum[label] / cluster_count[label]
        cluster_avg_distance.append((label, avg_distance))
        print(f"cluster {label} 的平均距离为: {avg_distance:.4f}")

    # ====== 排序并舍弃类内平均距离较大的类别 ======
    # 按照类内平均距离从小到大排序
    cluster_avg_distance.sort(key=lambda x: x[1])  # (label, avg_distance) 按 avg_distance 排序

    # 舍弃距离较大的类别，保留前 N 个类别
    top_n = 5  # 例如保留前 5 个聚类
    retained_clusters = set(label for label, _ in cluster_avg_distance[:top_n])
    print(f"保留的类别: {retained_clusters}")


    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        if label in retained_clusters: # 只保留在 retained_clusters 中的句子
            sentence_label_dict[label].append(sentence)         #同标签的放到一起

    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()

