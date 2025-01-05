#!/usr/bin/env python3
# coding: utf-8

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

# 加载文本
# 每行为一个句子
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

# 根据结果进行类内距离排序
def sort_by_intra_distance(sentences, vectors, kmeans):
    sentence_label_distance = []
    for sentence, vector, label in zip(sentences, vectors, kmeans.labels_):
        # 计算到该类中心的距离
        center = kmeans.cluster_centers_[label]
        distance = np.linalg.norm(vector - center)
        sentence_label_distance.append((sentence, label, distance))

    # 根据标签和距离进行排序
    sorted_results = defaultdict(list)
    for sentence, label, distance in sorted(sentence_label_distance, key=lambda x: (x[1], x[2])):
        sorted_results[label].append((sentence, distance))
    return sorted_results

def main():
    model = load_word2vec_model(r"D:\\AI\\pycharm20240103\\nlp\\第五周代码\\model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sorted_results = sort_by_intra_distance(list(sentences), vectors, kmeans)

    for label, sentences in sorted_results.items():
        print(f"cluster {label} :")
        for sentence, distance in sentences[:10]:  # 只打印前10个
            print(f"{sentence.replace(' ', '')} (distance: {distance:.4f})")
        print("---------")

if __name__ == "__main__":
    main()
