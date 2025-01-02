import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


'''
基于训练好的词向量模型进行聚类
聚类采用Kmeans算法
'''

# 输入模型文件路径
# 加载训练好的模型
def load_word2vec_model(model_path):
    model = Word2Vec.load(model_path)
    return model

# 将所有标题分词
def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

# 将语句向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

def main():
    model = load_word2vec_model("model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定个聚类数量", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算
    sentence_label_dict = defaultdict(list)
    # 实现基于kmeans结果类内距离的排序
    cluster_center = kmeans.cluster_centers_  # 获取所有聚类中心
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起
    # key:label, value:每个句子的词向量组成句子集合
    vector_label_dict = defaultdict(list)
    # key:label, value:cos距离
    cosine_distance_dict = defaultdict(list)
    for vector, label in zip(vectors, kmeans.labels_):  # 取出句子向量和标签
        vector_label_dict[label].append(vector)  # 同标签的放到一起

    for center, i in zip(cluster_center, range(0, n_clusters)):
        vector_label = vector_label_dict[i]
        cosine_distance = 0
        for vector in vector_label:
            # 计算每个聚类中心与每个向量的距离
            cosine_distance += 1 - np.dot(vector, center) / (np.linalg.norm(vector) * np.linalg.norm(center))
        cosine_distance = cosine_distance / len(vector_label)
        cosine_distance_dict[i] = cosine_distance
    # 按照距离排序
    sort_cosine_distance_dict = sorted(cosine_distance_dict.items(), key=lambda x: (x[1], x[0]))
    for label, cosine_distance in sort_cosine_distance_dict:
        print("cluster: %s, cosine_distance: %s" % (label, cosine_distance))
        for value in sentence_label_dict[label]:
            print(value.replace(" ", ""))  # 打印每个聚类的前10个标题
        print("---------")

    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):
    #         print(sentences[i].replace(" ", ""))  # 打印每个聚类的前10个标题
    #     print("---------")

if __name__ == "__main__":
    main()

