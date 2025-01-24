import math
from collections import defaultdict

import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans


# 加载模型
def load_model(path):
    model = Word2Vec.load(path)
    return model


# 获取文本
def get_sentences(path):
    sentences = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            sen = line.strip()
            sentences.add(" ".join(jieba.lcut(sen)))
    print("获取句子数量：", len(sentences))
    return sentences


# 将文本转换向量
def sentences_to_vector(sentences, model):
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
    model = load_model("model.w2v")
    sentences_result = get_sentences("titles.txt")
    sentences_vector = sentences_to_vector(sentences_result, model)

    n_clusters = int(math.sqrt(len(sentences_result)))
    km = KMeans(n_clusters)
    km.fit(sentences_vector)

    cluster_centers = km.cluster_centers_
    labels = km.labels_

    distans = [[] for _ in range(n_clusters)]
    for i in range(len(sentences_result)):
        label = labels[i]
        point = sentences_vector[i]
        center = cluster_centers[label]
        text = list(sentences_result)[i]
        # 计算欧几距离
        d = distance(point, center)
        distans[label].append((text, d))

    for c in range(n_clusters):
        print("cluster %s :" % c)
        distans[c].sort(key=lambda x: x[1], reverse=True)
        for s in range(min(10, len(distans[c]))):
            point, dd = distans[c][s]
            print(point.replace(" ", "") + "distance：%s" % dd)
        print("---------")

    # sentences_label = defaultdict(list)
    # for sen, label in zip(sentences_result, km.labels_):
    #     sentences_label[label].append(sen)
    #
    # for label, sentences in sentences_label.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

def distance(point, centerPoint):
    return np.sqrt(np.sum((point - centerPoint) ** 2))


if __name__ == '__main__':
    main()
