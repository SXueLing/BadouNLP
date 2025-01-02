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
    try:
        model = Word2Vec.load(path)
        return model
    except FileNotFoundError:
        print(f"模型文件 {path} 未找到。")
        return None


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
    unknown_words = set()
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        word_count = 0
        for word in words:
            try:
                vector += model.wv[word]
                word_count += 1
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                 vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


# 使用肘部法则选择最优的聚类数量
def select_optimal_clusters(vectors, max_k=10):
    sse = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(vectors)
        sse.append(kmeans.inertia_)
    # 简单的肘部法则实现，寻找SSE下降变缓的点
    diffs = [sse[i + 1] - sse[i] for i in range(len(sse) - 1)]
    optimal_k = np.argmax(diffs) + 1
    return optimal_k


def main():
    model_path = r"D:\python\model.w2v"
    sentence_path = "titles.txt"

    model = load_word2vec_model(model_path)
    if model is None:
        return

    sentences = load_sentence(sentence_path)
    vectors = sentences_to_vectors(sentences, model)

    # 使用肘部法则选择最优的聚类数量
    n_clusters = select_optimal_clusters(vectors)
    print("指定聚类数量：", n_clusters)

    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)

    # 计算每个聚类的类内距离
    intra_cluster_distances = []
    for cluster_id in range(n_clusters):
        cluster_mask = kmeans.labels_ == cluster_id
        cluster_vectors = vectors[cluster_mask]
        cluster_center = kmeans.cluster_centers_[cluster_id]
        distance = np.sum(np.linalg.norm(cluster_vectors - cluster_center, axis = 1))
        intra_cluster_distances.append(distance)

    # 对类内距离进行排序
    sorted_indices = np.argsort(intra_cluster_distances)

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起

    # 根据排序后的索引重新整理聚类结果
    sorted_sentence_label_dict = defaultdict(list)
    for new_index, old_index in enumerate(sorted_indices):
        sorted_sentence_label_dict[new_index] = sentence_label_dict[old_index]

    # 打印排序后的聚类结果
    for label, sentences in sorted_sentence_label_dict.items():
        print(f"cluster {label} :")
        for i in range(min(10, len(sentences))):
            print(sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
