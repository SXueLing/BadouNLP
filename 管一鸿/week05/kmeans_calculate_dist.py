#!/usr/bin/env python3  
# coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


# 输入模型文件路径
# 加载训练好的模型
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


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


# 计算质心平均距离（余弦相似度）
from sklearn.metrics.pairwise import cosine_similarity
def calculate_cosineSim_center(sentences,vector_center,model):
    vectors_of_current_sentences = sentences_to_vectors(sentences, model) # 获取当前质心句子的向量
    distance_sum = 0
    distance_of_single_vector = dict()
    vector_center = vector_center.reshape(1, -1) # 统一形状
    for i,vector in enumerate(vectors_of_current_sentences):
        vector = vector.reshape(1,-1)
        distance_of_single_vector[sentences[i]] = cosine_similarity(vector,vector_center)
        distance_sum += distance_of_single_vector[sentences[i]]
    
    return distance_sum / len(vectors_of_current_sentences),distance_of_single_vector

def main():
    model = load_word2vec_model(r"./model.w2v")  # 加载词向量模型
    sentences = load_sentence(r"./titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起
    centers = kmeans.cluster_centers_
    for label, sentences in sentence_label_dict.items():
        avg_distance_current_label, distance_dict = calculate_cosineSim_center(sentences,centers[label],model)
        print("cluster %s :" % label)
        print("avg distance : %f " % avg_distance_current_label)
        top_10_sentences = sorted(distance_dict.items(),key=lambda x : x[1])[:10]
        # for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
        #     print(sentences[i].replace(" ", ""))
        for sentence in top_10_sentences:
            print(sentence[0].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
