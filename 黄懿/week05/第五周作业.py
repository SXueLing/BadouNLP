# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
from collections import defaultdict

import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans


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


def distance_cal(p1, p2):
    sum = 0
    for i in range(len(p1)):
        sum += pow(p1[i] - p2[i], 2)
    return pow(sum, 0.5)


def main():
    model = load_word2vec_model("model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    vector_label_dict = defaultdict(list)
    for sentence, vector, label in zip(sentences, vectors, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起
        vector_label_dict[label].append(vector)
    # 计算类内平均距离
    distance_mean_per_cluster = list()
    for i in range(n_clusters):
        sum = 0.0
        for j in range(len(sentence_label_dict[i])):
            sum += distance_cal(kmeans.cluster_centers_[i], vector_label_dict[i][j])
        distance_mean_per_cluster.append([i, sum / len(sentence_label_dict[i])])
    # 取类内平均距离最近的前10个类
    target_num = 10
    distance_mean_per_cluster = sorted(distance_mean_per_cluster, key=lambda x:x[1])
    cluster_target = distance_mean_per_cluster[:target_num]
    for i in range(target_num):
        label = cluster_target[i][0]
        distance = cluster_target[i][1]
        print("cluster %s , 类内平均距离 %.3f :" % (label, distance))
        for j in range(min(10,len(sentence_label_dict[label]))):  # 随便打印几个，太多了看不过来
            print(sentence_label_dict[label][j].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
