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


def cosine_distance(vector1, vector2):
    """
    计算两个向量的余弦距离。

    参数:
        vector1 (numpy.ndarray): 第一个向量
        vector2 (numpy.ndarray): 第二个向量

    返回:
        float: 余弦距离
    """
    # 确保输入是 numpy 数组
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    # 计算向量的点积
    dot_product = np.dot(vector1, vector2)

    # 计算向量的模（范数）
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    # 计算余弦相似度
    cosine_similarity = dot_product / (norm1 * norm2)

    # 余弦距离 = 1 - 余弦相似度
    cosine_dist = 1 - cosine_similarity

    return cosine_dist



def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化
    # print("vectors",vectors)

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    centroids = kmeans.cluster_centers_
    # 打印质心坐标
    # print("质心坐标：")
    # print(centroids)
    # print(len(centroids))
    # print(kmeans.labels_)
    # print(sentences)
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")


    instances_dict = defaultdict(list)
    for vector, label  in zip(vectors, kmeans.labels_):
        instance=cosine_distance(vector,centroids[label])
        instances_dict[label].append(instance)
    result = {}
    for key, values in instances_dict.items():
        # print(key)
        # print(values)
        # print(len(values))

        if values:  # 确保列表不为空
            average = sum(values) / len(values)
        else:
            average = 0  # 如果列表为空，平均值为 0
        result[key] = average
    print(result)

    # 按值排序（升序）
    sorted_dict_by_value = {k: v for k, v in sorted(result.items(), key=lambda item: item[1])}

    print("余弦距离从小到大排序：")
    print(sorted_dict_by_value)



if __name__ == "__main__":
    main()

