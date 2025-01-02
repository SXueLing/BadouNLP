import numpy as np
import re
import jieba
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import math
from collections import defaultdict
from scipy.spatial.distance import euclidean


def load_vectors(vector_path):
    model = Word2Vec.load(vector_path)
    return model


def load_titles(title_path):
    sentences = []
    sentence_vectors = []
    with open(title_path, encoding="utf8") as f:
        for line in f:
            sentences.append(" ".join(jieba.cut(line.strip())))
    return sentences


# print(load_titles("titles.txt")[:10])

def sentences_to_vectors(sentences, model):
    sentence_vectors = []
    for sentence in sentences:
        words = sentence.split()  # 什么意思
        sentence_vector = np.zeros(model.vector_size)
        for word in words:
            try:
                sentence_vector += model.wv[word]
            except KeyError:
                sentence_vector += np.zeros(model.vector_size)
        sentence_vectors.append(sentence_vector / len(words))
    return sentence_vectors


# model = load_vectors("model.w2v")
# sentences = load_titles("titles.txt")
# sentence_vectors = sentences_to_vectors(sentences, model)
# print(len(sentences))
# print(len(sentence_vectors))

def cluster(sentence_vectors, n_cluster):
    kmeans = KMeans(n_cluster)
    kmeans.fit(sentence_vectors)
    return kmeans


def main():
    model = load_vectors("model.w2v")
    sentences = load_titles("titles.txt")
    sentence_vectors = sentences_to_vectors(sentences, model)
    n_cluster = int(math.sqrt(len(sentences)))
    kmeans = cluster(sentence_vectors, n_cluster)
    clusters = defaultdict(list)
    distance = defaultdict(list)
    for sentence_vector, sentence, label in zip(sentence_vectors, sentences, kmeans.labels_):
        clusters[label].append(sentence.replace(" ", ""))
        distance[label].append(euclidean(sentence_vector, kmeans.cluster_centers_[label]))
    for label, vector_lst in sorted(distance.items(), key=lambda item: np.mean(item[1]), reverse=True):
        print("===============cluster:%d==================" % label)
        print(np.mean(vector_lst))
        for sentence in clusters[label]:
            print(sentence)


if __name__ == "__main__":
    main()
