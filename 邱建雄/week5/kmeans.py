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
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化


    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算


    # 匹配标题，标题向量，label
    vector_label_dict = defaultdict(dict)
    for vector, label, sentence in zip(vectors, kmeans.labels_, sentences):
        vector_tuple = tuple(vector)
        vector_label_dict[label][sentence] = vector_tuple

    #根据label 获取中心点，并计算标题向量到中心点距离
    center_label_dict = defaultdict(dict)
    for label, sentence_dict in vector_label_dict.items():
        for sentence, vector_tuple in sentence_dict.items():
            vector = np.array(vector_tuple)
            center_label_dict[label][sentence] = np.linalg.norm(kmeans.cluster_centers_[label] - vector)  #欧式距离

    # 根据欧式距离从小到大排序，并输出前十个句子
    for label, sentence_center_len in center_label_dict.items():
        order_center_len = sorted(sentence_center_len.items(), key=lambda x: x[1])
        #print(order_center_len)
        print("cluster %s :" % label)
        for sentence, center_len in order_center_len[:10]:
            print(sentence.replace(" ", ""), center_len)
        print("---------")



if __name__ == "__main__":
    main()

