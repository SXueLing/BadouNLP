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
from scipy.spatial.distance import cdist

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

# 计算簇内的平均距离
def calculate_inertia(cluster_vectors):
    centroid = np.mean(cluster_vectors, axis=0)
    distances = cdist(cluster_vectors, [centroid])
    return np.mean(distances)

def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起

    # 计算每个簇的类内距离并排序
    cluster_inertia = {}
    for label, sentences in sentence_label_dict.items():
        cluster_vectors = vectors[kmeans.labels_ == label]
        inertia = calculate_inertia(cluster_vectors)
        cluster_inertia[label] = inertia

    # 根据类内距离排序
    sorted_clusters = sorted(cluster_inertia.items(), key=lambda x: x[1])

    for label, inertia in sorted_clusters:
        print(f"cluster {label} (inertia: {inertia}):")
        for i in range(min(10, len(sentence_label_dict[label]))):  #随便打印几个，太多了看不过来
            print(sentence_label_dict[label][i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()
