#coding: utf-8
"""
week5 homework: 实现基于kmeans结果类内距离的排序
12/26/2024

    ----洪扬
"""
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


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
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model("model.w2v")
    sentences = load_sentence("titles.txt")
    vectors = sentences_to_vectors(sentences, model)

    n_clusters = int(math.sqrt(len(sentences)))
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        sentence_label_dict[label].append(sentence)

    cluster_distances = []
    for label, sentences in sentence_label_dict.items():
        cluster_vectors = sentences_to_vectors(sentences, model)
        center = kmeans.cluster_centers_[label]
        distances = [np.linalg.norm(vec - center) for vec in cluster_vectors]
        avg_distance = np.mean(distances)
        cluster_distances.append((label, avg_distance))

    cluster_distances.sort(key=lambda x: x[1])

    for label, distance in cluster_distances:
        print(f"cluster {label} avg distance: {distance}")
        cluster_sentences = sentence_label_dict[label]
        for i in range(min(10, len(cluster_sentences))):
            print(cluster_sentences[i].replace(" ", ""))
        print("---------")

#向量余弦距离
def cosine_distance(vec1, vec2):
    vec1 = vec1 / np.sqrt(np.sum(np.square(vec1)))  #A/|A|
    vec2 = vec2 / np.sqrt(np.sum(np.square(vec2)))  #B/|B|
    return np.sum(vec1 * vec2)

#欧式距离
def eculid_distance(vec1, vec2):
    return np.sqrt((np.sum(np.square(vec1 - vec2))))


if __name__ == "__main__":
    main()
