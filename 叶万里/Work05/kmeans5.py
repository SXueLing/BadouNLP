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


# 计算类内平均距离的函数
def calculate_intra_cluster_distance(vectors, labels, n_clusters):
    intra_distances = []
    for cluster_id in range(n_clusters):
        cluster_vectors = [vectors[i] for i in range(len(vectors)) if labels[i] == cluster_id]
        if len(cluster_vectors) < 2:
            intra_distances.append(0)  # 只有一个点的类内距离设为0
            continue
        distances = []
        for i in range(len(cluster_vectors)):
            for j in range(i + 1, len(cluster_vectors)):
                dist = np.linalg.norm(cluster_vectors[i] - cluster_vectors[j])
                distances.append(dist)
        avg_distance = sum(distances) / (len(cluster_vectors) * (len(cluster_vectors) - 1) / 2)
        intra_distances.append(avg_distance)
    return intra_distances


def main():
    model = load_word2vec_model(r"F:\Desktop\work_space\badou\八斗课程\week5 词向量及文本向量\model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    # 计算类内平均距离
    intra_cluster_distances = calculate_intra_cluster_distance(vectors, kmeans.labels_, n_clusters)

    # 将距离和对应的标签组合起来
    distance_label_list = [(intra_cluster_distances[i], i) for i in range(len(intra_cluster_distances))]
    # 按照距离从小到大排序
    distance_label_list.sort(key=lambda x: x[0])
    # 选取距离最小的前五个类的标签
    top_five_labels = [label for _, label in distance_label_list[:5]]

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起

    for label in top_five_labels:
        print("cluster %s :" % label)
        sentences_in_cluster = sentence_label_dict[label]
        for i in range(min(10, len(sentences_in_cluster))):  # 随便打印几个，太多了看不过来
            print(sentences_in_cluster[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
