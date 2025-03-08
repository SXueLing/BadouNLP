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


# 计算两点间距
def calcu_distance(p1, p2):
    tmp = 0
    for i in range(len(p1)):
        tmp += pow(p1[i] - p2[i], 2)
    return pow(tmp, 0.5)


# 作业：计算类内距离，并排序
# TODO
def calcu_distance_and_sort(sentence_label_dict, kmeans, model):
    cluster_centers = kmeans.cluster_centers_  # 所有质心
    distance_dict = defaultdict(dict)

    for label, sentences in sentence_label_dict.items():
        # print("label=",label)
        for i in range(len(sentences)):
            vector = np.zeros(model.vector_size)
            wordList = sentences[i].split()  # sentences[i]是分好词的一句话
            for ele in wordList:
                try:
                    vector += model.wv[ele]
                except KeyError:
                    # 部分词在训练中未出现，用全0向量代替
                    vector += np.zeros(model.vector_size)
            vectors = (vector / len(wordList))  # sentences[i]的句向量
            cluster_center = cluster_centers[int(label)]
            distance = calcu_distance(vectors, cluster_center)  # 样本和质心的距离
            distance_dict[label][sentences[i]] = distance

    for label, sentences_distance in distance_dict.items():
        sorted_sentences_distance = sorted(sentences_distance.items(), key=lambda item: item[1])  # 排序，距离从小到大
        distance_dict[label] = sorted_sentences_distance

    return distance_dict


def main():
    model = load_word2vec_model(r".\model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起


    # 作业：计算类内距离，并排序
    # TODO
    distance_dict = calcu_distance_and_sort(sentence_label_dict, kmeans, model)
    # print(distance_dict)

    # 打印前 10个
    for label, sentences_distance_item in distance_dict.items():
        print("label=",label)
        for i in range(min(10, len(sentences_distance_item))): #随便打印几个，太多了看不过来
            print(sentences_distance_item[i][0].replace(" ", ""), sentences_distance_item[i][1])
        print("---------")

if __name__ == "__main__":
    main()
