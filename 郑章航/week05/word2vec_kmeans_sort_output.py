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
import numpy.linalg as LA
import matplotlib.pyplot as plt


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


def find_optimal_k(wcss):
    """
    通过肘部法则确定最优的k值
    这里简单通过比较相邻WCSS差值的变化来近似判断斜率变化，找到肘部位置
    """
    if len(wcss) < 3:
        return 1
    diff = [wcss[i] - wcss[i + 1] for i in range(len(wcss) - 1)]
    diff_diff = [diff[i] - diff[i + 1] for i in range(len(diff) - 1)]
    optimal_index = 0
    max_diff_diff = 0
    for index in range(len(diff_diff)):
        if diff_diff[index] > max_diff_diff:
            max_diff_diff = diff_diff[index]
            optimal_index = index + 1
    return optimal_index + 1

def main():
    model = load_word2vec_model(r"D:\BaiduNetdiskDownload\week5 词向量及文本向量\model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    wcss = []
    for k in range(1, 50):  # 尝试k从1到10
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(vectors)
        wcss.append(kmeans.inertia_)
    # 绘制肘部法则曲线
    # plt.plot(range(1, 100), wcss)
    # plt.title('Elbow Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('WCSS')
    # plt.show()

    # 通过肘部法则确定最优k值
    optimal_k = find_optimal_k(wcss)
    print("根据肘部法则确定的最优聚类数量:", optimal_k)

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    cluster_centers = kmeans.cluster_centers_
    print("聚类中心", cluster_centers)

    # 用于存储每个聚类内向量到中心点余弦相似度平均值
    cluster_avg_similarity_dict = defaultdict(float)
    sentence_label_dict = defaultdict(list)
    sentence_vector_dict = {}
    for sentence, vector in zip(sentences, vectors):
        label = kmeans.labels_[np.where((vectors == vector).all(axis=1))[0][0]]
        sentence_label_dict[label].append(sentence)
        sentence_vector_dict[sentence] = vector

        # 计算当前句子向量到对应聚类中心的余弦相似度
        center_vector = cluster_centers[label]
        cosine_similarity = np.dot(vector, center_vector) / (LA.norm(vector) * LA.norm(center_vector))
        # 累加当前聚类内的余弦相似度
        cluster_avg_similarity_dict[label] += cosine_similarity

    # 计算每个聚类内的平均余弦相似度
    for label in cluster_avg_similarity_dict:
        cluster_avg_similarity_dict[label] /= len(sentence_label_dict[label])

    # 将聚类按照平均余弦相似度进行排序
    sorted_cluster_labels = sorted(cluster_avg_similarity_dict, key=lambda x: cluster_avg_similarity_dict[x], reverse=True)

    for label in sorted_cluster_labels:
        print("cluster %s :" % label)
        print("平均余弦相似度:", cluster_avg_similarity_dict[label])
        sentences_in_cluster = sentence_label_dict[label]
        for sentence in sentences_in_cluster:
            print("标题:", sentence.replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()

