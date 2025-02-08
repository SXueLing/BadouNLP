#!/usr/bin/env python3
# coding: utf-8

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
import pandas as pd


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
    word_list = []
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
        word_list.append(sentence)
    return np.array(vectors), word_list


def euclidean_distance(a, b):
    """
    欧几里得距离
    :param a:
    :param b:
    :return:
    """
    return np.linalg.norm(a - b)


def main():
    model = load_word2vec_model(r"model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors, words_list = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    distance_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        index = words_list.index(sentence)
        vector = vectors[index]
        cluster_center = kmeans.cluster_centers_[label]
        distance = euclidean_distance(vector, cluster_center)
        distance_label_dict[label].append(distance)
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起

    mean_label_dict = {}
    for distance in distance_label_dict:
        all_list = distance_label_dict[distance]
        mean = np.mean(all_list)
        mean_label_dict[distance] = mean

    sorted_series = pd.Series(mean_label_dict).sort_values()
    sorted_keys = sorted_series.index.tolist()

    for label in sorted_keys:
        sentences = sentence_label_dict[label]
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
