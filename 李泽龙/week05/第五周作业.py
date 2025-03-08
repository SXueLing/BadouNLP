#kmeans算法（聚类）
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

#欧式距离计算
def calculate_distance(center_location, sentence_location):
    distance = np.linalg.norm(sentence_location - center_location)
    return distance
    
    

def main():
    model = load_word2vec_model(r"E:/nlp_learn/practice/week05/model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化
    # 计算每个类标签中每句话向量与中心质点的距离，并求平均
    average_distances = {}

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    #center_sentence_label_dict = defaultdict(list)
    for i, label in enumerate(kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(vectors[i])         #同标签的放到一起
    # for center_sentence, label in zip(kmeans.cluster_centers_, kmeans.labels_):  #取出中心句子和标签
    #     center_sentence_label_dict[label].append(center_sentence)    #一个标签对应一个中心句子
    for label in sentence_label_dict:
        #获取该标签的所有句子向量
        sentences_with_label = sentence_label_dict[label]
        #获取该标签的中心质点
        center = kmeans.cluster_centers_[label]
        
        #计算每句话与中心质点的距离
        distances = [calculate_distance(center,sentence) for sentence in sentences_with_label]
        
        #求平均距离
        average_distance = np.mean(distances)
        average_distances[label] = average_distance
        
    #依照平均距离排序
    sorted_average_distances = sorted(average_distances.items(), key = lambda x:x[1], reverse = True)
    print("每个类标签的平均距离：", average_distances)
    print("每个类标签的平均距离（排序后）：")
    for label, average_distance in sorted_average_distances:
        print(f"标签{label}:平均距离{average_distance}")
        
    # 获取排名前十的类标签
    top_ten_labels = [label for label, _ in sorted_average_distances[:5]]
    context_label_dict = defaultdict(list)
    for context, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        context_label_dict[label].append(context)         #同标签的放到一起
    # 打印排名前十的类标签和对应的所有句子
    print("排名前十的类标签和对应的所有句子：")
    for label in top_ten_labels:
        print(f"标签 {label}:")
        for sentence in context_label_dict[label]:
            print(sentence)
        print()  # 打印空行以分隔不同的标签
        
    
    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

if __name__ == "__main__":
    main()

