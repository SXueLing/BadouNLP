
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
from scipy.spatial.distance import cdist
from scipy.spatial import distance

# 定义两个点
# point1 = [1, 2]
# point2 = [4, 6]
 
# # 计算两点之间的欧氏距离
# euclidean_distance = distance.euclidean(point1, point2)

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding='utf8') as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    # 获取句子数量： 1796
    # 指定聚类数量： 42
    return sentences


#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split() #sentence是分好词的，空格分开
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
    # Word2Vec<vocab=19322, vector_size=128, alpha=0.025> model
    sentences = load_sentence('titles.txt') #加载所有标题
    vectors = sentences_to_vectors(sentences, model) #将所有标题向量化
    print(vectors, 'vectors')
    n_clusters = int(math.sqrt(len(sentences))) #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters) #定义一个kmeans计算类
    kmeans.fit(vectors) # 进行聚类计算
    print(len(kmeans.labels_), 'kmeans.labels_') #1796 kmeans.labels_
    print(len(sentences), 'sentences') # 1796
    # [21 29 38 ... 33 33 20] kmeans.labels_ 每个句子对应分类数值
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_): #取出句子和标签
        sentence_label_dict[label].append(sentence) #同标签的放到一起
    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        cluster_centers = kmeans.cluster_centers_[label] # 当前分类的质心
        # 根据欧式距离排序
        sentences = sorted(sentences, key = lambda x: distance.euclidean(cluster_centers, sentences_to_vectors([" ".join(jieba.cut(x))], model).squeeze())) # 根据欧氏距离排序
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")


    # te = '中国最美的地方:湖北的隐匿仙境'
    # print([" ".join(jieba.cut(te))], '中国最美的地方:湖北的隐匿仙境')
    # print(sentences_to_vectors([" ".join(jieba.cut(te))], model).squeeze())
    # print(len(kmeans.cluster_centers_), '质心')
    # print(kmeans.cluster_centers_.shape, '质心shape') # (42, 128)
    # print(cdist(vectors, kmeans.cluster_centers_, 'euclidean').shape, 'inertia_')
    # 1. cluster_centers_:最终聚类中心的坐标；
    # 2. labels_:每个样本点对应的类别标签；
    # 3. inertia_:每个样本点到距离它们最近的类中心的距离平方和，即SSE；
    # 4. n_iter:实际的迭代次数；
    # 5. n_features_in_:参与聚类的特征个数；
    # 6. feature_names_in_:参与聚类的特征名称。
    # print(vectors.shape, kmeans.cluster_centers_.shape) # (1796, 128) (42, 128)
    # print(len(sentence_label_dict.items())) # 42
    

   



 
if __name__ == "__main__":
    main()
