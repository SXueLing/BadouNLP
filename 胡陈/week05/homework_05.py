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
        for i,line in enumerate(f):
            sentence = line.strip()
            sentence=" ".join(jieba.cut(sentence))
            # if i<5:
            #     print(sentence)
            sentences.add(sentence)
        # for i,sentence in enumerate(sentences):
        #     if i < 5:
        #         print(sentence)

    print("获取句子数量：", len(sentences))
    
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for i,sentence in enumerate(sentences):
        words = sentence.split()  #sentence是分好词的，空格分开
        # words是一个句子列表
        # if i < 5:
        #     print(words)
        #     print(type(words))
            # print(model.vector_size)#128
        vector = np.zeros(model.vector_size)
        # print(f"original vector:{vector}")
        #所有词的向量相加求平均，作为句子向量
        for i,word in enumerate(words):
            try:
                vector += model.wv[word]
                # if i<1:
                #     print(f"vector:{vector}")
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    # print(f"np.array(vectors):{np.array(vectors)}")
    return np.array(vectors)


def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for i,sentence, label in zip(range(0,len(sentences)),sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
        # print(f"lable={label}")

    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

    # 计算每个点到其聚类中心的距离
    distances = []
    for i, point in enumerate(vectors):
        cluster_center = kmeans.cluster_centers_[kmeans.labels_[i]]
        distance = np.linalg.norm(point - cluster_center)#计算欧式距离
        distances.append((i, distance))

    # 排序
    sorted_distances = sorted(distances, key=lambda x: x[1])

    # 输出排序结果
    for i,distance in enumerate(sorted_distances):
        if i < 30:#只打印了前30个的类内距离
            index, dis = distance
            print(f"点 {index} 的类内距离为 {dis}")

if __name__ == "__main__":
    main()

