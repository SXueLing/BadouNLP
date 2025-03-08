# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
import os

os.environ["OMP_NUM_THREADS"] = "8"


# 输入模型文件路径
# 加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


def load_sentence(path):
    sentences = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子的向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


# 计算向量之间的欧式距离
def eculid_distance(vec1, vec2):
    distance = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return distance


def main():
    model = load_word2vec_model("F:/NLP/NLP-05/model.w2v")  # 加载词向量模型
    sentences = load_sentence("F:/NLP/NLP-05/titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)

    # 计算类内距离
    len_dict = defaultdict(list)
    for vector_index, label in enumerate(kmeans.labels_):
        vector = vectors[vector_index]  # 某句话的向量
        center = kmeans.cluster_centers_[label]  # 对应的类别中心向量
        distance = eculid_distance(vector, center)  # 计算欧式距离
        len_dict[label].append(distance)
    # 对于每一类，将类内所有文本到中心的欧式距离取平均
    for label, distance_list in len_dict.items():
        len_dict[label] = np.mean(distance_list)
    len_dict_order = sorted(len_dict.items(), key=lambda x: x[1], reverse=True)

    # 按照余弦距离顺序取出
    for label, distance_avg in len_dict_order:
        print("cluster %s, avg distance %f:" % (label, distance_avg))
        sentences = sentence_label_dict[label]
        for i in range(min(10, len(sentences))):
            print(sentences[i].replace(" ", ""))
        print("---------------")


if __name__ == '__main__':
    main()
