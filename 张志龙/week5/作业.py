#!/usr/bin/env python3  
#coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

"""
作业：Kmeans基础上实现按照类内距离排序
计算类内距离：欧式距离 / 余弦距离
"""

# 输入模型文件路径
# 加载训练好的模型
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

# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model("./badou/week5/model.w2v") # 加载词向量模型
    sentences = load_sentence("./badou/week5/titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)   # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)          # 进行聚类计算

    sentence_label_dict = defaultdict(list)  # defaultdict(list)的作用是创建一个带有默认值的字典，字典的值是列表
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)         # 同标签的放到一起

    # todo 计算类内距离：欧式距离 / 余弦距离
    distance_dict = defaultdict(list)  # 存储每个类别的距离
    print(kmeans.labels_)  # 打印每个句子的标签
    print(kmeans.cluster_centers_)  # 打印每个类别的中心点
    for index, label in enumerate(kmeans.labels_):
        # 计算每个句子和其所在类别的中心点的距离
        distance_dict[label].append(euclidean_distance(vectors[index], kmeans.cluster_centers_[label]))
    
    # 计算每个类别的平均距离
    for label,distance_list in distance_dict.items():
        distance_dict[label] = np.mean(distance_list)

    # 按照类内距离排序
    sorted_distance_dict = sorted(distance_dict.items(), key=lambda x: x[1], reverse=True) # 默认是升序，reverse=True表示降序
    print(sorted_distance_dict)

    for label, distances in sorted_distance_dict:  # 遍历每个类别
        print("cluster %s ,平均距离为 %f :" % (label, distances))
        sentences = sentence_label_dict[label]  # 获取该类别的所有句子
        for i in range(min(10, len(sentences))):   # 随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

# 计算两个向量的欧氏距离
def euclidean_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))

# 计算两个向量的余弦距离
def cosine_distance(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return 1 - (dot_product / (norm_a * norm_b))

if __name__ == "__main__":
    main()

