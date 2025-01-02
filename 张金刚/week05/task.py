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

from sklearn.metrics.pairwise import cosine_similarity

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
    model = load_word2vec_model(r".\model.w2v") #加载词向量模型
    sentences = list(load_sentence("titles.txt"))#加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化
    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算
    # 计算每个样本到所有簇中心的距离

    kmeans_centers= kmeans.cluster_centers_ # 获取每个簇的中心点
    sentence_label_dict = defaultdict(list)
    # for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
    #     sentence_label_dict[label].append(sentence)         #同标签的放到一起
    for index, (sentence, label) in enumerate(zip(sentences, kmeans.labels_)):  # 取出句子和标签
        sentence_label_dict[label].append((index, sentence))  # 同标签的放到一起，附带索引信息

    for label, sentences_with_index in sentence_label_dict.items():
        print("cluster %s :" % label)
        # print("簇中心:", kmeans_centers[label])
        # 计算每个句子向量与簇中心的相似度
        similarities = []
        for index, sentence in sentences_with_index:
            sentence_vector = vectors[index]
            similarity = cosine_similarity([sentence_vector], [kmeans_centers[label]])[0][0]
            distance = 1 - similarity  # 计算余弦距离
            similarities.append((similarity, distance, index, sentence))

            # 输出排序后的相似度和句子
            # 按照相似度排序
        # similarities.sort(reverse=True, key=lambda x: x[0])
        # 按照余弦距离升序排序
        similarities.sort(key=lambda x: x[1])
        # 计算平均余弦距离
        avg_distance = sum(distance for _, distance, _, _ in similarities) / len(similarities)
        print(f"平均余弦距离: {avg_distance:.4f}")

        # 输出最相似的前10个句子及距离
        for _, distance, index, sentence in similarities[:10]:
            print(f"距离: {distance:.4f}.  {sentence.replace(' ', '')}")

        print("---------")

        # for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
        #     print(sentences[i].replace(" ", ""))
        # print("---------")

if __name__ == "__main__":
    main()

