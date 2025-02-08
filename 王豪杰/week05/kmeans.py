# coding: utf-8

import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


def load_sentences(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print(f"获取句子数量：{len(sentences)}")
    return sentences


def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence 是已经分好词的，空格分开
        vector = np.zeros(model.vector_size)
        valid_word_count = 0
        for word in words:
            if word in model.wv:
                vector += model.wv[word]
                valid_word_count += 1
        # 如果有效词汇数大于0，则计算平均值
        if valid_word_count > 0:
            vector /= valid_word_count
        vectors.append(vector)
    return np.array(vectors)


def cosine_distance(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 > 0 and norm2 > 0:
        return np.dot(vec1, vec2) / (norm1 * norm2)
    return 0.0


def compute_density(kmeans, vectors):
    density_dict = defaultdict(list)
    for vector_index, label in enumerate(kmeans.labels_):
        vector = vectors[vector_index]
        center = kmeans.cluster_centers_[label]
        distance = cosine_distance(vector, center)
        density_dict[label].append(distance)

    for label, distance_list in density_dict.items():
        density_dict[label] = np.mean(distance_list)

    return density_dict


def main():
    model_path = "model.w2v"
    sentences_path = "titles.txt"

    model = load_word2vec_model(model_path)  
    sentences = load_sentences(sentences_path) 
    vectors = sentences_to_vectors(sentences, model)  
  
    n_clusters = int(math.sqrt(len(sentences)))
    print(f"指定聚类数量：{n_clusters}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(vectors)

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        sentence_label_dict[label].append(sentence)

    density_dict = compute_density(kmeans, vectors)

    density_order = sorted(density_dict.items(), key=lambda x: x[1], reverse=True)

    for label, distance_avg in density_order:
        print(f"cluster {label}, avg distance {distance_avg:.6f}:")
        sentences_in_cluster = sentence_label_dict[label]
        for i in range(min(5, len(sentences_in_cluster))):  
            print(sentences_in_cluster[i].replace(" ", ""))
        print("=====================")


if __name__ == "__main__":
    main()
