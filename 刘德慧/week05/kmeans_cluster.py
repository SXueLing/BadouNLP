# !/usr/bin/env python3
# coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
import os
# import re
# import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
from scipy.spatial import distance


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

    model_path = os.path.join(os.path.dirname(__file__), 'model.w2v')
    sentence_path = os.path.join(os.path.dirname(__file__), 'titles.txt')
    model = load_word2vec_model(model_path)  # 加载词向量模型
    sentences = load_sentence(sentence_path)  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    # If no argument is given, the constructor creates a new empty list.
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起

    distances = defaultdict(list)
    for sen_idx, label in enumerate(kmeans.labels_):
        d = distance.euclidean(vectors[sen_idx],
                               kmeans.cluster_centers_[label])
        distances[label].append(d)

    for label, ds_list in distances.items():
        distances[label] = np.mean(ds_list)
        # 将字典按照value值升序排序，sorted()返回的是一个列表
    ds_order = sorted(distances.items(), key=lambda item: item[1])
    # new_ds = ds_order[:len(distances) - 5]  # 去掉比较分散的的五个类

    # count = 0
    for label, distance_avg in ds_order:
        # count += 1
        print("cluster %s , avg distance %f: " % (label, distance_avg))
        sentences = sentence_label_dict[label]
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")
    # print(count)


if __name__ == "__main__":
    main()
