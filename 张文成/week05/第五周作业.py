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
    model = load_word2vec_model(r"E:\BaiduNetdiskDownload\八斗精品课nlp\第五周 词向量\week5 词向量及文本向量\week5 词向量及文本向量\model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化
    max_show = 5 #每一簇打印前几个

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    distance_per_point = {}
    sentence_distance_per_group = [[] for _ in range(n_clusters)]
    sentence_label_dict = defaultdict(list)
    for sentence, label, vector in zip(sentences, kmeans.labels_, vectors):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
        distance = sum((vector - kmeans.cluster_centers_[label])**2)  #计算每一个点距离中心点的距离
        # distance_per_point[distance] = sentence    # 把每个点的结果储存在distance_per_point{6:文本1, 9:文本2, 8:文本3, 16:文本4.....}
        distance_per_point[sentence] = distance    # 把每个点的结果储存在distance_per_point{文本1:6, 文本2:9, 文本3:8, 文本4:16.....}
        sentence_distance_per_group[label].append((sentence, distance))   #同标签的放到一起 [[(文本1,6), (文本2,9),...] .... [(文本3:8), (文本4,16)...]]
        # sentence_distance_per_group[label].append((sentence, distance))


        # for sentence, text_distance, label in zip(sentences, distance_per_point, kmeans.labels_):
    #     sentence_distance_per_group[label].append((distance_per_point[text_distance],text_distance))         #同标签的放到一起 [[(文本1,6), (文本2,9),...] .... [(文本3:8), (文本4,16)...]]

    for label, group in enumerate(sentence_distance_per_group):  # [(文本1,6), (文本2,9),...]
        sorted_group = sorted(group, key=lambda x: x[1])     # [(文本1,6), (文本2,9),...]
        print("cluster %s :" % label)
        for j in range(min(max_show, len(sorted_group))):
            print(sorted_group[j][0])    #  





if __name__ == "__main__":
    main()

