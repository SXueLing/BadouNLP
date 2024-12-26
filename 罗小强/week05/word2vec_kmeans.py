#!/usr/bin/env python3  
# coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
from collections import defaultdict

import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import os
# 设置环境变量以避免KMeans的内存泄漏问题
os.environ['OMP_NUM_THREADS'] = '8'


# 输入模型文件路径
# 加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


def load_sentence(path):
    sentences = []
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.append(" ".join(jieba.cut(sentence)))
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
    model = load_word2vec_model(r"model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)  # 使用defaultdict来避免KeyError
    # 计算到质心点的距离
    n_clusters = kmeans.cluster_centers_  # 42X128的矩阵，代表聚类中心
    print("聚类中心：", n_clusters.shape)
    # 采用欧式距离计算质心点到句子的距离
    for i, vector in enumerate(vectors):
        distance = np.linalg.norm(vector - n_clusters[kmeans.labels_[i]])
        sentence_label_dict[kmeans.labels_[i]].append((distance, sentences[i]))  # 距离和句子对应起来

    # for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
    #     sentence_label_dict[label].append(sentence)         #同标签的放到一起
    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        sentences.sort(key=lambda x: x[0])  # 按距离排序
        for i in range(min(10, len(sentences))):
            distance, sentence = sentences[i]
            #for distance, sentence in sentences:
            print(f"{distance:.3f} {sentence.replace(' ', '')}")  # 去掉空格
        print("---------")
        # for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
        #     print(sentences[i].replace(' ', ''))
        # print("---------")


if __name__ == "__main__":
    main()

# 输入模型文件路径
# 加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


def load_sentence(path):
    sentences = []
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.append(" ".join(jieba.cut(sentence)))
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
    model = load_word2vec_model(r"model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)  # 使用defaultdict来避免KeyError
    # 计算到质心点的距离
    n_clusters = kmeans.cluster_centers_  # 42X128的矩阵，代表聚类中心
    print("聚类中心：", n_clusters.shape)
    # 采用欧式距离计算质心点到句子的距离
    for i, vector in enumerate(vectors):#遍历句子向量：使用 enumerate 获取每个句子向量的索引和值
        distance = np.linalg.norm(vector - n_clusters[kmeans.labels_[i]])  # 计算质心点到句子向量的距离
        sentence_label_dict[kmeans.labels_[i]].append((distance, sentences[i]))  # 距离和句子对应起来
    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        sentences.sort(key=lambda x: x[0])  # 按距离排序
        for i in range(min(10, len(sentences))):
            distance, sentence = sentences[i]
            print(f"{distance:.3f} {sentence.replace(" ", "")}")  # 去掉空格
        print("---------")


if __name__ == "__main__":
    main()
