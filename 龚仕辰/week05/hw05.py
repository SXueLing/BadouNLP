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

def calc_cos_distance(vector1, vector2): 
    v1 = vector1 / np.linalg.norm(vector1)
    v2 = vector2 / np.linalg.norm(vector2)
    return np.dot(v1, v2)

def calculate_intra_cluster_distance(cluster_vectors):
    if len(cluster_vectors) == 0:
        return 0
    centroid = np.mean(cluster_vectors, axis=0)
    distances = [calc_cos_distance(vec, centroid) for vec in cluster_vectors]
    return np.mean(distances)

def main():
    model = load_word2vec_model('model.w2v') #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    # TODO：实现基于kmeans结果类内距离的排序
    n_clusters = int(math.sqrt(len(sentences)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(vectors)
    kmeans.fit(vectors)

    # 获取每个簇的向量
    clusters = defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(vectors[i])

    # 计算每个簇的类内距离
    intra_cluster_distances = {}
    for label, cluster_vectors in clusters.items():
        intra_cluster_distances[label] = calculate_intra_cluster_distance(cluster_vectors)

    # 按类内距离排序
    sorted_clusters = sorted(intra_cluster_distances.items(), key=lambda x: x[1], reverse=True)

    # 输出排序结果
    for label, distance in sorted_clusters:
        print(f"簇 {label}: 类内距离 {distance}")

if __name__ == "__main__":
    main()

