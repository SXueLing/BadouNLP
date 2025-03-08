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

#计算余弦距离
def cosine_distance(vec1, vec2):
    # 计算点积
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    # 计算余弦相似度
    cosine_similarity = dot_product / (norm1 * norm2)
    # 计算余弦距离
    cosine_distance = 1 - cosine_similarity
    return cosine_distance

def main():
    model = load_word2vec_model(r"C:\work\BaDou\model\week05\model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起

    distance_dict = defaultdict(list)
    # 计算所有类内距离
    for vector_index, label in enumerate(kmeans.labels_):
        # 获取向量
        vector = vectors[vector_index]
        # 对应类别的中心向量
        cluster_centers = kmeans.cluster_centers_[label]
        # 计算余弦距离
        distance = cosine_distance(vector, cluster_centers)
        distance_dict[label].append(distance)

    for label, distances in distance_dict.items():
        # 计算每个类别平均距离
        distance_dict[label] = np.mean(distances)

    # 按照平均距离降序排序
    distance_order_dict = sorted(distance_dict.items(), key=lambda x: x[1], reverse=True)
    print(distance_order_dict)

    # 按照余弦距离顺序输出
    for label, distance_avg in distance_order_dict:
        print("cluster %s , 平均距离 %f: " % (label, distance_avg))
        # 当前类别所有句子
        sentences = sentence_label_dict[label]
        for i in range(min(10, len(sentences))):
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()

