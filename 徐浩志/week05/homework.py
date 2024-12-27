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
    sentences = list()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.append(" ".join(jieba.cut(sentence)))
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
    model = load_word2vec_model(r"/Users/xuhaozhi/PycharmProjects/pythonProject/AI/week5/原始版本/model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起

    centers = kmeans.cluster_centers_
    # 使用一个字典，key为 label，value为距离的列表
    Distance_Dict = {x:[] for x in range(n_clusters)}
    Distance = lambda x, y: np.sqrt(np.sum((np.array(x) - np.array(y)) ** 2))
    # 循环聚类数量次
    for i in range(n_clusters):
        #找到每一个类的向量
        for each_sentence in sentence_label_dict[i]:
            # 找到向量
            vector_index = sentences.index(each_sentence)
            vector = vectors[vector_index]
            # 求距离
            vector_distance = Distance(vector, centers[i])
            Distance_Dict[i].append(vector_distance)
    for key in Distance_Dict:
        Distance_Dict[key] = np.mean(Distance_Dict[key])
    Distance_list = [Distance_Dict[key] for key in Distance_Dict]
    Distance_list.sort()

    for cluster_close_five in range(5):
        for key in Distance_Dict:
            distance1 = Distance_list[cluster_close_five]
            distance2 = Distance_Dict[key]
            if distance1 == distance2:
                print("This is cluster: {}, distance is {}".format(key, Distance_Dict[key]))
                for i in range(10):
                    print(sentence_label_dict[key][i].replace(' ',''))

    # 这一段代码是简单的检测 cluster_centers_是否和 label 按顺序一一对应
    # Distance = lambda x, y: np.sqrt(np.sum((np.array(x) - np.array(y)) ** 2))
    # min_distance_index = []
    # for i in range(n_clusters):
    #     Distances = []
    #     for each_center in centers:
    #         x = sentences.index(sentence_label_dict[i][0])
    #         x = vectors[x]
    #         y = each_center
    #         # print(x, y, sep='\n')
    #         Distances.append(Distance(x, y))
    #     print("This is {} turn, {}".format(i, Distances))
    #     min_distance_index.append(np.argmin(Distances))
    # print(min_distance_index)


    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

if __name__ == "__main__":
    main()

