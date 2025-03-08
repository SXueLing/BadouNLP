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
    # 加载训练好的词向量模型
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    # 加载句子数据
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            # 使用jieba进行分词，并将分词后的句子加入集合
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size) #初始化一个全0向量,vetor_size是词向量的维度。
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word] #词向量相加，wv 是该模型中的一个属性，通常表示“词向量”（word vectors），model.wv[word] 就是获取该单词对应的词向量。
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
    kmeans.fit(vectors)          #进行聚类计算，vectors是所有标题的向量表示
    #调用 fit 方法，将句子的向量表示作为输入，执行聚类分析。fit 方法会训练模型，并将每个句子分配给一个聚类,返回一个 KMeans 对象，该对象包含有关聚类分析的所有信息。
    #kmeans.labels_是每个标题对应的聚类标签，
    #kmeans.cluster_centers_是每个聚类的中心点向量表示
    #kmeans.inertia_是每个聚类内所有点的向量表示与中心点向量表示的欧氏距离之和,即每个聚类的内聚度,越小越好,
    #kmeans.n_iter_是kmeans算法迭代次数,
    #kmeans.n_clusters_是聚类数量,即n_clusters,这个属性是只读的,不能修改,如果修改了,会报错,所以需要用n_clusters属性来获取聚类数量

    sentence_label_dict = { i:[] for i in range(n_clusters)}
    
    for sentence,vector, label in zip(sentences,vectors, kmeans.labels_):
        distance = np.linalg.norm(kmeans.cluster_centers_[label] - vector) #计算每个标题与聚类中心的距离
        sentence_label_dict[label].append((sentence, distance)) #将标题和距离加入对应的聚类中
    for label , sentence  in sentence_label_dict.items():
        sentence_label_dict[label].sort(key=lambda x: x[1]) #按照距离从小到大排序
        print("cluster %s :" % label)
        for i in range(min(10, len(sentence))):
            print(sentence[i][0].replace(" ", ""))
        print("---------")

    # sentence_label_dict = defaultdict(list)
    # for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
    #     sentence_label_dict[label].append(sentence)         #同标签的放到一起
    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

if __name__ == "__main__":
    main()

