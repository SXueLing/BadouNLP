import math
import re
import json
import jieba
import numpy as np
import os
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

script_dir = os.path.dirname(__file__)

def load_sentences(path):
    sentences = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            sentences.add(' '.join(jieba.cut(line)))
        print(f'句子数量：{len(sentences)}')
    return sentences

def sentence_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)
                

def main():
    model = Word2Vec.load(os.path.join(script_dir, 'model.w2v'))
    sentences = load_sentences(os.path.join(script_dir, 'titles.txt'))
    vectors = sentence_to_vectors(sentences, model)
    n_clusters = int(math.sqrt(len(sentences)))
    print(f'聚类数量：{n_clusters}')
    # 定义KMeans类
    kmeans = KMeans(n_clusters)
    # 进行聚类计算
    kmeans.fit(vectors)
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        sentence_label_dict[label].append(sentence)
    
    # 获取聚类中心的列表
    centers = kmeans.cluster_centers_

    # 平均距离字典
    avg_dict = {}
    for label, cluster_sentences in sentence_label_dict.items():
        distance = 0
        for cluster_sentence in cluster_sentences:
            # 找到该句子在句子向量中的位置，然后计算欧式距离
            distance += np.linalg.norm(vectors[list(sentences).index(cluster_sentence)] - centers[label])
        # 计算平均距离
        avg_distance = distance / len(cluster_sentences)
        # 保存平均距离
        avg_dict[label] = avg_distance
    # 按照平均距离排序
    sorted_avg_dict = sorted(avg_dict.items(), key=lambda item: item[1])
    
    # 按照类内平均距离由小到大，输出每一类的句子
    for label, avg_distance in sorted_avg_dict:
        print(label)
        for sentence_label, cluster_sentences in sentence_label_dict.items():
            if label == sentence_label:
                for i in range(min(10, len(cluster_sentences))):
                    print(cluster_sentences[i].replace(" ", ""))
        print('--------------------------')

if __name__ == '__main__':
    
    main()
