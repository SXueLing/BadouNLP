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
    # python3.7以后默认set多次循环取值取出的值的顺序都相同
    # 但是这里需要转换成list，方便获取每个句子对应的位置信息
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


# 计算两个向量之间的余弦距离
def cosine_similarity(v1, v2):
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# 对字典按照值的大小进行排序，这里为降序排序
def sort_dict_by_key(dictionary):
    # 将字典的项转换为列表
    items = list(dictionary.items())
    # 对列表进行排序，这里按照值进行降序排序
    items.sort(key=lambda x: x[1], reverse=True)
    # 将排序后的列表转换回字典
    sorted_dict = dict(items)
    return sorted_dict


def main():
    model = load_word2vec_model(r"E:\code\python\test\pythonProject\AI_learning\self\week5\model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentences.index(sentence))         #同标签的放到一起，存放句子对应的位置信息
    # 最终字典（排序之后）
    sentence_label_dict_final = defaultdict(list)
    # 类内距离的排序
    for label, sentence_index_list in sentence_label_dict.items():
        distance_dict = {}
        # 拿到lable向量
        label_vector = kmeans.cluster_centers_[label]
        # 获取每个句向量到lable向量的距离
        for i, sentence_index in enumerate(sentence_index_list):
            sentence_vector = vectors[sentence_index]
            distance = cosine_similarity(label_vector, sentence_vector)
            # 每个句向量对应的跟中心向量的距离
            distance_dict[sentence_index] = distance
        # 对获取到的距离字典进行排序
        distance_dict = sort_dict_by_key(distance_dict)
        # 将每个lable排序后的句子逐一添加进最终字典里面
        for sentence_index, distance in distance_dict.items():
            sentence_label_dict_final[label].append(sentences[sentence_index])
    # 打印字典输出
    for label, sentences in sentence_label_dict_final.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()

