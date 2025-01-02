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

    sentence_label_dict = defaultdict(list)
    # 同时记录每个句子对应的向量，方便后续计算距离
    sentence_vector_dict = {}
    for sentence, vector in zip(sentences, vectors):
        label = kmeans.labels_[np.where((vectors == vector).all(axis=1))[0][0]]
        sentence_label_dict[label].append(sentence)
        sentence_vector_dict[sentence] = vector

    for label, sentences_in_cluster in sentence_label_dict.items():
        print("cluster %s :" % label)
        # 获取当前聚类的中心向量
        center_vector = cluster_centers[label]
        distance_sentence_list = []
        for sentence in sentences_in_cluster:
            vector = sentence_vector_dict[sentence]
            # 计算余弦相似度，通过1 - 余弦相似度得到类似距离的衡量值
            cosine_similarity = np.dot(vector, center_vector) / (LA.norm(vector) * LA.norm(center_vector))
            distance = 1 - cosine_similarity
            distance_sentence_list.append((distance, sentence))

        # 按照距离从小到大排序
        sorted_distance_sentence_list = sorted(distance_sentence_list, key=lambda x: x[0])

        for distance, sentence in sorted_distance_sentence_list:
            print("标题:", sentence.replace(" ", ""), " 余弦夹角值:", distance)
        print("---------")

if __name__ == "__main__":
    main()

