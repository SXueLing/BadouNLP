# coding: utf-8

import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


def load_model(filepath):
    word2vec_model = Word2Vec.load(filepath)
    return word2vec_model


def get_sentences(filepath):
    unique_sentences = set()
    with open(filepath, encoding="utf8") as file:
        for line in file:
            trimmed = line.strip()
            tokenized = " ".join(jieba.cut(trimmed))
            unique_sentences.add(tokenized)
    print("加载的句子数量:", len(unique_sentences))
    return unique_sentences


def vectorize_sentences(sentences, model):
    sentence_vectors = []
    for sentence in sentences:
        tokens = sentence.split()
        avg_vector = np.zeros(model.vector_size)
        for token in tokens:
            try:
                avg_vector += model.wv[token]
            except KeyError:
                avg_vector += np.zeros(model.vector_size)
        avg_vector /= len(tokens)
        sentence_vectors.append(avg_vector)
    return np.array(sentence_vectors)


def compute_cosine_similarity(vector_a, vector_b):
    norm_a = vector_a / np.linalg.norm(vector_a)
    norm_b = vector_b / np.linalg.norm(vector_b)
    return np.dot(norm_a, norm_b)


def compute_euclidean_distance(vector_a, vector_b):
    return np.linalg.norm(vector_a - vector_b)


def main():
    w2v_model = load_model("model.w2v")
    titles = get_sentences("titles.txt")
    title_vectors = vectorize_sentences(titles, w2v_model)

    cluster_count = int(math.sqrt(len(titles)))
    print("聚类数量:", cluster_count)
    kmeans_model = KMeans(n_clusters=cluster_count)
    kmeans_model.fit(title_vectors)

    clusters = defaultdict(list)
    for title, label in zip(titles, kmeans_model.labels_):
        clusters[label].append(title)

    cluster_density = defaultdict(list)
    for idx, label in enumerate(kmeans_model.labels_):
        vector = title_vectors[idx]
        center = kmeans_model.cluster_centers_[label]
        similarity = compute_cosine_similarity(vector, center)
        cluster_density[label].append(similarity)
    
    for label, similarities in cluster_density.items():
        cluster_density[label] = np.mean(similarities)
    
    sorted_clusters = sorted(cluster_density.items(), key=lambda x: x[1], reverse=True)

    for label, avg_sim in sorted_clusters:
        print(f"聚类 {label}，平均余弦相似度: {avg_sim:.6f}")
        sample_titles = clusters[label]
        for title in sample_titles[:10]:
            print(title.replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
