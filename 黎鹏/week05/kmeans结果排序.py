from collections import defaultdict
import math
from sklearn.cluster import KMeans
import numpy as np

def main():
    model = load_word2vec_model("model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个KMeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label, vector in zip(sentences, kmeans.labels_, vectors):
        sentence_label_dict[label].append((sentence, vector))  # 同标签的放到一起

    for label, sentence_vector_pairs in sentence_label_dict.items():
        # 计算每个向量到该聚类中心的距离
        cluster_center = kmeans.cluster_centers_[label]
        sentence_vector_pairs.sort(key=lambda x: np.linalg.norm(x[1] - cluster_center))

        print(f"cluster {label}:")
        for i in range(min(10, len(sentence_vector_pairs))):  # 打印距离中心最近的10个句子
            print(sentence_vector_pairs[i][0].replace(" ", ""))
        print("---------")
