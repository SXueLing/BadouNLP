import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

def load_model(path):
    model = Word2Vec.load(path)
    return model


def load_corpus(path):
    corpus = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            sentence = line.strip()
            corpus.add(" ".join(jieba.lcut(sentence)))
    return corpus


def corpus_vector(corpus, model):
    vectors = []
    for sentence in corpus:
        vector = np.zeros(model.vector_size)
        words = sentence.split()
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_model('word2vec.w2v')
    corpus = load_corpus('titles.txt')
    vectors = corpus_vector(corpus, model)

    cluster_num = int(math.sqrt(len(corpus)))
    kmeans = KMeans(cluster_num)
    kmeans.fit(vectors)
    distances = kmeans.transform(vectors)  # 计算每个句子到每个聚类中心的距离
    min_distances = distances.min(axis=1)   # 取最小的距离即为该句子到所属的聚类中心的距离
    kmeans_dict = defaultdict(list)

    for sentence, label in zip(corpus, kmeans.labels_):
        kmeans_dict[label].append(sentence)  # 将句子和所属的聚类中心编号对应起来


    distances_dict = defaultdict(list)
    for sentence, distance in zip(corpus, min_distances):
        for key, values in kmeans_dict.items():
            if sentence in values:
                distances_dict[key].append([distance, sentence])  # 将句子和到所属的聚类中心的距离对应起来
    for key, values in distances_dict.items():
        sorted_values = sorted(values, key=lambda x: x[0])  # 按照距离排序
        print("label序号：%s" % key)
        for i in range(min(10, len(sorted_values))):
            print(sorted_values[i][1].replace(' ', ''))
        print("----------------------------------")
        
if __name__ == "__main__":
    main()
