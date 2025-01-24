# coding: utf-8
import math

import jieba
# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


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
    print(f"获取句子数量：{len(sentences)}")
    return sentences


# 文本向量化
def sentence_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 每出现过的词，用全0向量代替
                vector += np.zeros(model.vector_size)
            vectors.append(vector / len(words))
    return np.array(vectors)


# 计算label: avg_distance的结果，使用欧式距离
# vectors: 句子向量，如100个句子
# labels: 分类标签，如100个句子，就有100个标签
# centers: 中心向量，10个标签对应10个中心坐标
def cal_avg_distance(vectors, labels, centers):
    # 1、首先计算出每个点到中心的距离，并归到相应的label下: label: [0.2, 0.3...]
    label_cluster_distance_dict = defaultdict(list)
    for label, vector in zip(labels, vectors):
        center = centers[label]
        distance = np.linalg.norm(vector - center)
        label_cluster_distance_dict[label].append(distance)
    # 2、计算每个label下的平均距离：label: 0.5
    return {label: np.mean(distance_list) for label, distance_list in label_cluster_distance_dict.items() }


def main():
    # 加载词向量模型
    model = load_word2vec_model(r"model.w2v")
    sentences = load_sentence("titles.txt")
    # 全部标题向量化
    vectors = sentence_to_vectors(sentences, model)

    # 指定聚类数量
    n_clusters = int(math.sqrt(len(vectors)))
    print(f"聚类数量为：{n_clusters}")

    # 定义KMeans计算类
    kmeans = KMeans(n_clusters)
    # 聚类计算
    kmeans.fit(vectors)

    label_sentence_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        label_sentence_dict[label].append(sentence)
    # for label, sentences in sentence_label_dict.items():
    #     print(f"cluster {label}")
    #     for i in range(min(10, len(sentences))):
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

    # todo 计算平均距离，取出前10的cluster
    label_avg_distance_dict = cal_avg_distance(vectors, kmeans.labels_, kmeans.cluster_centers_)
    sorted_clusters = sorted(label_avg_distance_dict.items(), key=lambda x: x[1])
    top_10_clusters = sorted_clusters[:10]
    print(label_avg_distance_dict)
    print(top_10_clusters)
    print(f"距离最小的前十个cluster：")
    for label, avg_distance in top_10_clusters:
        print(f"label: {label}，平均距离：{avg_distance}")
        items = label_sentence_dict[label]
        for i in range(min(10, len(items))):
            print(items[i].replace(" ", ""))
        print("---------")


if __name__ == '__main__':
    main()

