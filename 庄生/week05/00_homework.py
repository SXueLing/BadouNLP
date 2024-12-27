# !/usr/bin/env python3  
# coding: utf-8

# 基于04案例中训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
import pprint

# 输入模型文件路径
# 加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    # print(model.wv["学生"])   # 就能得到"学生"对应的向量
    # print(model)   # Word2Vec<vocab=19322, vector_size=100, alpha=0.025>
    # print(model.vector_size)   # 100
    # print(np.zeros(model.vector_size))   # [0. 0. 0. ....0. 0.] 100个0的数组
    # print(np.zeros(model.vector_size)+model.wv["学生"])   # 得到学生向量
    return model

load_word2vec_model("model.w2v")


def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    return sentences
# load_sentence("titles.txt")


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        # model.vector_size是词向量的维度
        # np.zeros(model.vector_size) 为：[0. 0. 0. ....0. 0.] 100个0的数组
        vector = np.zeros(model.vector_size)  # 生成model.vector_size个0
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，未出现会报异常所以捕获异常，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def getDistance(p1, p2):
    """
    计算两个数据点之间的欧几里得距离

    :param p1: 第一个数据点
    :param p2: 第二个数据点
    :return: 两个数据点之间的欧几里得距离
    """
    tmp = 0
    for i in range(len(p1)):
        tmp += pow(p1[i] - p2[i], 2)  # 计算坐标差值的平方和
    return pow(tmp, 0.5)  # 开方得到欧几里得距离


def main():
    # 第1步、加载模型
    model = load_word2vec_model("model.w2v")
    # 第2步、加载所有标题
    sentences = load_sentence("titles.txt")
    print(sentences)
    # 第3步、将所有标题向量化
    vectors = sentences_to_vectors(sentences, model)

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)          # 进行聚类计算

    sentence_label_dict = defaultdict(list)  # 创建值为列表的字典

    # kmeans.cluster_centers_ 为向量中心点数组
    # print(kmeans.cluster_centers_, len(kmeans.cluster_centers_))

    # sentences 是所有句子集合
    # len(sentences) 为1796
    # kmeans.labels_ 是所有句子对应的类别列表，也有1796项 [14  0 40 ... 30 17  9]
    for sentence, label, vector in zip(sentences, kmeans.labels_, vectors):  # 取出句子和标签
        sentence_label_dict[label].append({
            'vector': vector,
            'sentence': sentence
        })         # 同标签的放到一起

    res_dict = {}
    # 组织数据：
    for cluster_center, label, label_classified_list in zip(kmeans.cluster_centers_, sentence_label_dict, sentence_label_dict.values()):

        for item in label_classified_list:
            item['distance'] = getDistance(cluster_center, item['vector'])
            del item['vector']

        res_dict[label] = sorted(label_classified_list, key=lambda x: x['distance'])

    # pprint.pprint(res_dict)

    # 展示：
    for label, d_list in res_dict.items():
        print(f'---------排序后，类别为{label}的数据有：---------')
        for item in d_list:
            newitem = item['sentence'].replace("   ", "&*&*&*&").replace(" ", "").replace("&*&*&*&", " ")
            print(newitem)



if __name__ == "__main__":
    main()

