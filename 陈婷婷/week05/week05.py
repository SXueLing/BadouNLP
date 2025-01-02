#coding: utf-8


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

def sentences_to_vetors(sentences, model):
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
    vectors = sentences_to_vetors(sentences, model)

    n_clusters = int(math.sqrt(len(sentences)))
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)

    sentence_label_dict = defaultdict(list)
    in_cluster_distances = []
    for label, center in enumerate(kmeans.cluster_centers_):
        cluster_points = vectors[kmeans.labels_ == label]
        dis = np.linalg.norm(cluster_points - center, axis = 1)
        in_cluster_dis = np.mean(dis)
        in_cluster_distances.append((label, in_cluster_dis))
    in_cluster_distances.sort(key=lambda x: x[1])
    for label, dis in in_cluster_distances:
        print(f"cluster {label} 的类内距离: {dis:.4f}")
    #for sentence, label in zip(sentences, kmeans.labels_):
    #    sentence_label_dict[label].append(sentence)
    #for label, sentence in sentence_label_dict.items():
    #    print("cluster %s :" % label)
    #    for i in range(min(10, len(sentence))):
    #        print(sentence[i].replace(" ", ""))
    #    print("-----------")

if __name__ == "__main__":
    main()
