#!/usr/bin/env python3
#coding: utf-8

# 使用get_sorted_within_cluster_distance函数实现类内平均距离排序

import math
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


def get_sorted_within_cluster_distance(vectors, kmeans) -> dict:
    """计算类内平均距离，返回从小到大排序的字典:{类别：类内平均距离}"""
    distances = kmeans.transform(vectors)
    labels = kmeans.labels_
    # 存储每个类型的类内距离总和以及类内样本数
    res_dict = defaultdict(dict)
    for i in range(len(distances)):
        if not res_dict.get(labels[i]):
            res_dict[labels[i]] = {"distance": 0, "point_num": 0}
        res_dict[labels[i]]["distance"] += distances[i][labels[i]]
        res_dict[labels[i]]["point_num"] += 1
    res_dict = {k: v["distance"]/v["point_num"] for k, v in res_dict.items()}
    return dict(sorted(res_dict.items(), key=lambda x:x[1]))

def main():
    model = load_word2vec_model(r"E:\日常\学习\八斗\第五周 词向量\week5 词向量及文本向量\model.w2v") #加载词向量模型
    sentences = load_sentence("E:\日常\学习\八斗\第五周 词向量\week5 词向量及文本向量\\titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化
    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    res = get_sorted_within_cluster_distance(vectors, kmeans)
    print(res)
    for label, distance in res.items():
        # 按距离排序打印各类的一些句子
        print("cluster %s :" % label)
        print("within_cluster_distance :{}".format(distance))
        sentences = sentence_label_dict[label]
        for i in range(min(10, len(sentences))): 
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()
