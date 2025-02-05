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
    model = load_word2vec_model(r"/data/xmx/code/nlp/week5/model.w2v") #加载词向量模型
    sentences = load_sentence("/data/xmx/code/nlp/week5/titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    for type in ['cos_similarity', 'euclidean_distance', 'manhattan_distance', 'mahalanobis_distance']:
        sentence_label_dict = defaultdict(list)
        distance_dict = defaultdict(list)
        for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
            sentence_vectors = sentences_to_vectors(sentences, model)   #将标题向量化
            center = kmeans.cluster_centers_[label]
            #计算句子和中心余弦距离
            if type == 'cos_similarity':
                cos_similarity = np.dot(sentence_vectors, center) / (np.linalg.norm(sentence_vectors) * np.linalg.norm(center))
                distance_dict[label].append(cos_similarity)
            #计算句子和中心欧式距离
            elif type == 'euclidean_distance':
                euclidean_distance = np.linalg.norm(sentence_vectors - center)
                distance_dict[label].append(euclidean_distance)
            #计算句子和中心曼哈顿距离
            elif type == 'manhattan_distance':
                manhattan_distance = np.sum(np.abs(sentence_vectors - center))
                distance_dict[label].append(manhattan_distance)
            #计算句子和中心马氏距离
            elif type == 'mahalanobis_distance':
                mahalanobis_distance = np.sqrt(np.dot(np.dot((sentence_vectors - center), np.linalg.inv(np.cov(sentence_vectors))), (sentence_vectors - center).T))
                distance_dict[label].append(mahalanobis_distance)
            sentence_label_dict[label].append([sentence,label])         #同标签的放到一起

        avg_distance = {}
        for label in distance_dict.keys():
            # 每类计算平均距离，排序
            avg_distance[label] = np.mean(distance_dict[label])
        avg_distance = sorted(avg_distance.items(), key=lambda x: x[1], reverse=False)
        print("聚类数量：", n_clusters)
        print("距离计算方式：", type)
        print(avg_distance)
        print("---------")

    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

if __name__ == "__main__":
    main()

