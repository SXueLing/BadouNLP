import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

# 输入模型文件路径，加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

# 加载句子并分词
def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

# 将句子转化为向量
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence 是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

# 基于类内距离对聚类结果进行排序
def sort_by_cluster_distance(sentences, vectors, labels, cluster_centers):
    cluster_dict = defaultdict(list)
    for sentence, vector, label in zip(sentences, vectors, labels):
        # 计算当前句子到对应聚类中心的欧式距离
        distance = np.linalg.norm(vector - cluster_centers[label])
        cluster_dict[label].append((sentence, distance))
    
    # 对每个聚类内的句子按距离从小到大排序
    sorted_clusters = {}
    for label, items in cluster_dict.items():
        sorted_clusters[label] = sorted(items, key=lambda x: x[1])  # 按距离排序
    return sorted_clusters

# 主函数
def main():
    model = load_word2vec_model("./model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个 KMeans 计算类
    kmeans.fit(vectors)  # 进行聚类计算

    # 获取排序后的聚类结果
    sorted_clusters = sort_by_cluster_distance(
        sentences, vectors, kmeans.labels_, kmeans.cluster_centers_
    )
    
    for label, items in sorted_clusters.items():
        print("Cluster %s:" % label)
        for sentence, distance in items[:10]:  # 只打印前10个
            print(f"{sentence.replace(' ', '')} (距离: {distance:.4f})")
        print("---------")

if __name__ == "__main__":
    main()
