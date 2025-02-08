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
from sklearn.metrics.pairwise import cosine_distances
import heapq
from operator import itemgetter

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
        #print('词向量维度====='+str(model.vector_size))
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
    model = load_word2vec_model(r"D:\八斗\课件\第五周 词向量\week5 词向量及文本向量\model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    ####################计算聚类类内平均距离，并按照距离升序，打印出前五的聚类和类内部分句子 start###########

    
    # 获取聚类中心
    cluster_centers = kmeans.cluster_centers_

    # 获取每一个句子的所属标签
    cluster_labels = kmeans.labels_

    sentence_label_dict = defaultdict(list)
    vector_label_dict   = defaultdict(list)
    for sentence,vector, label in zip(sentences,vectors, cluster_labels):  #取出句子和标签
        sentence_label_dict[label].append(sentence)       #同标签的放到一起
        vector_label_dict[label].append(vector)           #同标签的放到一起

    distance_label_dict   = defaultdict(list)
    for (label, sentences),(label2,vectors) in  zip(sentence_label_dict.items(),vector_label_dict.items()):
        euclidean_distance = 0
        for i in range(len(vectors)):  #循环聚类所有点 用于计算每一个聚类的点到质心距离
            euclidean_distance += np.linalg.norm(vectors[i] - cluster_centers[label])# 计算欧式距离 的和
        euclidean_distance/len(vectors) # 聚类点到质心距离平均值
        distance_label_dict[label].append(euclidean_distance)
    
    # 使用 heapq.nsmallest 找到前 5 个最小值项
    sorted_data = dict(heapq.nsmallest(5, distance_label_dict.items(), key=itemgetter(1)))


    #循环sorted_data
    for label, distance in sorted_data.items():
        print("cluster %s 聚类类内平局距离 %s :" %(label,distance))
        sentences = sentence_label_dict[label]
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
             print(sentences[i].replace(" ", ""))
        print("---------")
       

if __name__ == "__main__":
    main()

