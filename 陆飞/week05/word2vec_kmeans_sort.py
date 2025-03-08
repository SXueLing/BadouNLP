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

from sklearn.metrics import euclidean_distances


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


def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    # 获取质心
    centers = kmeans.cluster_centers_
    # 创建一个字典来存储每个类别的向量及其距离
    clustered_vectors = {i: [] for i in range(kmeans.n_clusters)}

    # 计算每个向量到其质心的距离，并存储
    for sentence, vector, label in zip(sentences, vectors, kmeans.labels_):
        distance = euclidean_distances([vector], [centers[label]])[0][0]
        clustered_vectors[label].append((sentence, vector, distance))

    # 对每个类别的向量按距离排序
    for label in clustered_vectors:
        clustered_vectors[label].sort(key=lambda x: x[2])

    # 打印排序后的结果
    for label in clustered_vectors:
        print("----------------------------------")
        print(f"Cluster {label}:")
        for sentence, vector, distance in clustered_vectors[label]:
            print(f"  Sentence: {sentence.replace(' ', '')}, Distance to center: {distance:.4f}")

if __name__ == "__main__":
    main()

