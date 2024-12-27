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
    return list(sentences)

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

def calculate_distance(vector, center): #计算欧几里得距离
    sum_distance = 0
    for x1, x2 in zip(center, vector):
        sum_distance += math.pow(x1-x2, 2)
    return math.sqrt(sum_distance)

def main():
    model = load_word2vec_model(r"D:\BaiduNetdiskDownload\八斗NLP课程\week5 词向量及文本向量\model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起

    distance = defaultdict(float)
    for label, sentence_list in sentence_label_dict.items(): #取出label和对应的句子
        center = kmeans.cluster_centers_[label]              #取出对应的聚心
        tmp = 0
        for sentence in sentence_list:
            idx = sentences.index(sentence)
            vector = vectors[idx]
            tmp += calculate_distance(vector, center)
        distance[label] = tmp / len(sentence_list)          #计算类内平均距离
    res = sorted(distance.items(), key = lambda item:item[1]) #根据平均距离从小到大排序

    for i in range(10):                                     #打印前10个最小平均距离聚心的类和对应的5条句子
        label = res[i][0]
        print("类别为: ", label)
        for similar_sentence in sentence_label_dict[label][:5]:
            print("".join(similar_sentence.split()))
        print("--------")

if __name__ == "__main__":
    main()

