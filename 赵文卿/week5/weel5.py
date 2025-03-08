'''
Author: Zhao
Date: 2025-01-01 10:19:28
LastEditTime: 2025-01-01 18:29:19
FilePath: word2vec_kmeans.py
Description: 基于训练好的词向量模型进行聚类
            聚类采用Kmeans算法

'''
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
            sentences.add(" ".join(jieba.cut(sentence=sentence)))
    #print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

def main():
    model = load_word2vec_model(r"week5/model/model.w2v") #加载词向量模型
    sentences = load_sentence("week5/file/titles.txt") #加载所有标题
    vectors = sentences_to_vectors(sentences, model) #将所有标题向量化

    print("每句话的矩阵大小",vectors.shape)
    
    n_clusters = int(math.sqrt(len(sentences))) #指定聚类数量
    #print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters=n_clusters) #定义一个kmeans计算类
    kmeans.fit(vectors) #进行聚类计算


    """
        以下为进行修改的地方，需实现：
            1.聚类结束后计算类内平均距离
            2.排序后，舍弃类内较低的类别

    """
    # 计算类内平均距离
    cluster_avg = []
    # 获取簇标签和簇中心
    for i,center in zip(kmeans.labels_,kmeans.cluster_centers_):
        members = vectors[kmeans.labels_ == i]
        distances = np.linalg.norm(members - center, axis=1)
        avg_distance = distances.mean()
        cluster_avg.append((i, avg_distance))

    #排序类内平均距离
    #print(cluster_avg)
    cluster_avg.sort(key=lambda x: x[1]) # 按类内平均距离从低到高排序
    #print("="*100)

    # 舍弃类内距离较高的类别
    threshold = 1.5 # 可根据实际需求调整阈值 
    filtered_clusters = [i for i, avg_distance in cluster_avg if avg_distance <= threshold]
    
    print("排序后的类内平均距离: ", cluster_avg)
    print("舍弃后的: ", filtered_clusters)

    filtered_vectors = np.vstack([vectors[kmeans.labels_ == i] for i in filtered_clusters])
    print("筛选后的矩阵形状: ", filtered_vectors.shape)
   
    return
    
    # sentence_label_dict = defaultdict(list)
    # for sentence, label in zip(sentences, kmeans.labels_): #取出句子和标签
    #     sentence_label_dict[label].append(sentence)         #同标签的放到一起
    # for label, sentence in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentence))):
    #         print(sentence[i].replace(" ",""))
    #     print("-"*100)



if __name__ == "__main__":
    main()
