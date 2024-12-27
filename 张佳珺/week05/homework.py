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


def main():
    #model = load_word2vec_model(r"F:\Desktop\work_space\badou\八斗课程\week5 词向量及文本向量\model.w2v")  # 加载词向量模型
    model = load_word2vec_model(r"model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    #读取每个类的质心坐标向量
    center_vectors = kmeans.cluster_centers_
    distances_sum = defaultdict(list) #存储每一类内所有点与质心距离
    for label, vector in zip(kmeans.labels_, vectors):
        dis = np.linalg.norm(vector - center_vectors[label])
        distances_sum[label].append(dis)
    dis_mean = {} #存储每类的组内距离均值
    for label, dis in distances_sum.items():
        dis_mean[label] = np.mean(dis)
    #打印组内距离最大的5类及其包含的句子
    dis1 = sorted(dis_mean.items(), key=lambda x:x[1])
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    for label, dis in dis1[0:6]:
        print("cluster %s - %f:" % (label,dis))
        # print(sentence_label_dict[label])
        for i in range(min(10,len(sentence_label_dict[label]))):
            print(sentence_label_dict[label][i].replace(" ", ""))
        print("---------")

    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

if __name__ == "__main__":
    main()

