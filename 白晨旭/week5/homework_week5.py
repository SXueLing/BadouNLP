#!/usr/bin/env python3
# coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法并按类内距离排序
import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


# 输入模型文件路径
# 加载训练好的模型
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


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model(r"C:\Users\chenxubai\Downloads\week5 词向量及文本向量\model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters, random_state=42)  # 定义一个kmeans计算类，设置随机种子以确保结果可复现
    kmeans.fit(vectors)  # 进行聚类计算

    # 计算每个样本到所有簇中心的距离
    distances = kmeans.transform(vectors)  # distances[i][j] 是第i个样本到第j个簇中心的距离

    sentence_label_dict = defaultdict(list)
    for idx, (sentence, label) in enumerate(zip(sentences, kmeans.labels_)):
        distance_to_center = distances[idx][label]  # 获取样本到其所属簇中心的距离
        sentence_label_dict[label].append((sentence, distance_to_center))  # 存储句子及其距离

    for label, sentence_distance_list in sentence_label_dict.items():
        print("cluster %s :" % label)

        # 按距离排序，升序为从最接近簇中心到最远
        sorted_sentences = sorted(sentence_distance_list, key=lambda x: x[1])

        for i in range(min(10, len(sorted_sentences))):  # 随便打印几个，太多了看不过来
            sentence, distance = sorted_sentences[i]
            print(f"{sentence.replace(' ', '')} (Distance: {distance:.4f})")
        print("---------")


if __name__ == "__main__":
    main()
