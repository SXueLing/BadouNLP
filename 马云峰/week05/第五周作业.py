import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

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

# 计算每个点到其所属簇中心的距离
def compute_intra_distances(vectors, labels, centers):
    distances = []
    for vector, label in zip(vectors, labels):
        center = centers[label]
        distance = np.linalg.norm(vector - center)
        distances.append(distance)
    return np.array(distances)

def main():
    model = load_word2vec_model(r"F:\Model\model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    centers = kmeans.cluster_centers_  # 获取簇中心
    intra_distances = compute_intra_distances(vectors, kmeans.labels_, centers)  # 计算类内距离

    # 按类内距离排序
    sorted_indices = np.argsort(intra_distances)
    sorted_sentences = [sentences[i] for i in sorted_indices]
    sorted_distances = intra_distances[sorted_indices]

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sorted_sentences, kmeans.labels_[sorted_indices]):  # 取出句子和标签
        sentence_label_dict[label].append((sentence, sorted_distances[sorted_sentences.index(sentence)]))  # 同标签的放到一起

    for label, items in sentence_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(items))):  # 随便打印几个，太多了看不过来
            sentence, distance = items[i]
            print(f"{sentence.replace(' ', '')} (Distance: {distance:.4f})")
        print("---------")

if __name__ == "__main__":
    main()
