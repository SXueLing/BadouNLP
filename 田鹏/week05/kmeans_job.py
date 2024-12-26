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
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算
    
    
    '''
    作业思路
    cluster_centers  (42,128)  42个类和128的向量
    kmeans.cluster_centers 得到42*128的矩阵 就是42个中心点的位置
    算出每一句话vectors(向量)和每一个中心点的距离 欧式距离 余弦距离
    计算平均类内距离
    按照类内平均距离，查看比较短的类内平均距离是否更相似
    labels_ 每句话所属的类别
    '''
    tags_vector_dict = defaultdict(list)
    coredata = np.array(kmeans.cluster_centers_)
    for vectors, label in zip(vectors, kmeans.labels_):  #取出词向量和标签
        tags_vector_dict[label].append(vectors)         #同标签的放到一起
    # print('0',len(sentence_label_dict[0]),sentence_label_dict[0][len(sentence_label_dict[0]-1)])  # 第0类，句子向量，一个有多少句子

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起

    # 计算质点到类内向量之间的距离
    vec_label_dict = defaultdict(list)  # 定义类别的距离的数据
    for dim in range(len(coredata)):
        vec1 = coredata[dim]  # 取出第i类的向量
        for j in range(len(tags_vector_dict[dim])):  # 循环第i类向量对应的所有句子长度
            vec2 = tags_vector_dict[dim][j]   # 查询第i类向量对应的第j个句子的向量
            vec1_exc = np.array(vec1)            # 第i类对应的数组
            vec2_exc = np.array(vec2)            # 查询第i类向量对应的第j个句子的向量的数组
            # dict = np.sqrt(np.sum(np.square(vec1_exc - vec2_exc)))   # 计算欧式距离
            dict = 1 - np.dot(vec1_exc, vec2_exc) / (np.linalg.norm(vec1_exc) * np.linalg.norm(vec2_exc))  # 计算余弦距离
            vec_label_dict[dim].append(dict)  #将第i类对应的j类句子循环添加入字典，用i作为键，对应的句子长度作为值

    # print(vec_label_dict[0])
    # 计算每个类别中对应的平均距离
    dim_dict = {}  # 每个类中的平均距离
    for i in range(n_clusters):
        all_num = 0
        len_num = len(tags_vector_dict[i])
        for j in vec_label_dict[i]:
            all_num += j
        dim_dict[i] = all_num/len_num
    # 每个维度的平均距离
    # print(dim_dict)  {0: 0.4988900664548514, 1: 0.646374459370966, 2: 0.47973769245700254, 3: 0.6629204827485263,
    # 每个维度对应的句子距离
    # print(vec_label_dict)   {0: [0.6010559058756749, 0.44001772432899555, 0.42305373066713753, 0.7397042569576312,
    # 每个维度对应的句子数据
    # print(sentence_label_dict) {21: ['双色球 头奖 爆 6 注 713 万奖池 3.5 亿   广东 1 人中 1427 万',
    result_label_dict = defaultdict(list)
    # 舍弃类中低于平均距离的数据
    for i, j in dim_dict.items():
        # print(i, j)  0 0.693044064883478
        for len_vec in range(len(vec_label_dict[i])):
            if vec_label_dict[i][len_vec] > j:
                result_label_dict[i].append(sentence_label_dict[i][len_vec])


    # print(dim_dict)
    # print(vec_label_dict)


    # print('1',len(sentence_label_dict[1]))
    # # 聚类结束后计算每个向量到中心点的距离
    for label, sentences in result_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()

