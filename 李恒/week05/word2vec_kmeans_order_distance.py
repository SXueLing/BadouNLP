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


def euclidean_distance(vec1, vec2):
    """计算两个向量的欧式距离"""
    return np.linalg.norm(np.array(vec1) - np.array(vec2))


def cosine_distance(vec1, vec2):
    """计算两个向量的余弦距离"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return 1 - (dot_product / (norm_vec1 * norm_vec2))


def main():
    model = load_word2vec_model(r"model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict, sentence_distance_dict = defaultdict(list), defaultdict(list)

    # calc distance
    for label, vector in zip(kmeans.labels_, vectors):
        # 两句距离结果不一样呢？？
        distance = euclidean_distance(vector, kmeans.cluster_centers_[label])
        # distance = 1 - cosine_distance(vector, kmeans.cluster_centers_[label])
        sentence_distance_dict[label].append(distance)
    sentence_distance = [{'label': label, 'distance': sum(v)/len(v)} for label, v in sentence_distance_dict.items()]
    sentence_distance.sort(key=lambda x: x['distance'])
    print({value['label']:value['distance'] for value in sentence_distance})

    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起

    for distance_info in sentence_distance:
        label = distance_info['label']
        distance = distance_info['distance']
        sentences = sentence_label_dict[label]
        print(f"cluster {label}, distance: {distance} :")
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
