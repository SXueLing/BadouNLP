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
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    # print(type(sentences))
    sentences_list = [item for item in sentences]
    # print(sentences_list,"transform sentences")
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    centers = kmeans.cluster_centers_
    # print(len(centers))
    # print(len(vectors))
    # print(len(kmeans.labels_))
    top_n = 10
    sentence_sort_dict = defaultdict(list)
    for index, label in enumerate(kmeans.labels_):
        distance = np.linalg.norm(vectors[index] - centers[label]) # L2范数即欧氏距离
        sentence_sort_dict[label].append((distance, sentences_list[index]))

    # print(sentence_sort_dict)

    for label in sentence_sort_dict:
        sentence_sort_dict[label].sort()
        sentence_sort_dict[label] = [sentence for _, sentence in sentence_sort_dict[label][:top_n]]

    for label, sentences in sentence_sort_dict.items():
        print(f"语句类别 : {label} 提取{top_n}份信息如下:")
        for sentence in sentences:
            item = sentence.replace(" ", "")
            print(f"\t{item}")

if __name__ == "__main__":
    main()

