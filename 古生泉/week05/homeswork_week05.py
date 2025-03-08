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
import pandas as pd

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

# 欧式距离
def Euclidean_distance(distance1,distance2):
    return np.sqrt(np.sum(np.square(distance1-distance2)))


def main():
    model = load_word2vec_model(r"D:\Python\PythonAi\pythonAi\test_week05\model.w2v") #加载词向量模型
    sentences = load_sentence(r"D:\Python\PythonAi\pythonAi\test_week05\titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化
    # print("vector_size:",model.vector_size)
    # print("vector_size_zero:",np.zeros(model.vector_size))
    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算


    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(5, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

# 以下为计算类内平均距离排序(用欧式距离)
    work_sentence_label_dict = defaultdict(list)
    for work_vector, work_label in zip(vectors, kmeans.labels_):
        work_sentence_label_dict[work_label].append(work_vector)


    # print(np.array(work_sentence_label_dict.keys()))
    cluster_centers = kmeans.cluster_centers_
    ltrip_distance = dict()
    for work_label,work_vector in work_sentence_label_dict.items():
        label_distance = []
        cluster_center = cluster_centers[work_label]
        for work_vect in work_vector:
            distance = Euclidean_distance(cluster_center,work_vect)
            label_distance.append(distance)
        avg_distance = sum(label_distance)/len(label_distance)
        ltrip_distance[work_label] = avg_distance
    
    sorted_items = sorted(ltrip_distance.items(), key=lambda item: item[1])
    for i ,j in sorted_items:
        print(f"第{i}中心,类内距离(欧式距离计算):{j}")
        

if __name__ == "__main__":
    main()


