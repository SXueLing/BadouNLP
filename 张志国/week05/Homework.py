import numpy as np
import jieba
import gensim
from gensim.models import Word2Vec
import math
from sklearn.cluster import KMeans
from collections import defaultdict

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法

def load_model():
    # 加载词向量模型
    model = Word2Vec.load('model.w2v')
    return model

def load_sentence():
    sentences = set()
    # 加载句子
    with open('titles.txt', 'r', encoding='utf-8') as f:
        for line in f:
            sentence = line.strip()
            words = jieba.lcut(sentence)
            sentences.add(' '.join(words))
    print("获取句子数量：", len(sentences))
    return sentences

def sentence2vec(model, sentences):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            if word in model.wv:
                vector += model.wv[word]
        vectors.append(vector / len(words))
    return np.array(vectors)
        
def main():
    # 加载词向量模型
    model = load_model()
    # 加载句子
    sentences = load_sentence()
    # 将句子转化为词向量
    vectors = sentence2vec(model, sentences)
    # 使用k-means算法对词向量进行聚类
    kmeans = KMeans(n_clusters=int(math.sqrt(len(sentences))))
    kmeans.fit(vectors)
    # 输出聚类结果
    sentence_label_dict = defaultdict(list)
    dis_label_dict = defaultdict(list)
    center = kmeans.cluster_centers_
    for sentence, vector, label in zip(sentences, vectors, kmeans.labels_):
        sentence_label_dict[label].append(sentence)
        # 计算每个点到聚类中心的距离
        dis_label_dict[label].append(np.linalg.norm(vector - center[label]))
    # 根据每个聚类中句子的平均距离进行排序
    sorted_list = sorted(dis_label_dict.items(), key=lambda x: sum(x[1]) / len(x[1]))[:5]
    for label, score in sorted_list:
        print(f'Cluster {label}: - {sum(score) / len(score):.2f}')
        sentence_list = sentence_label_dict[label]
        for i in range(min(10, len(sentence_list))):
            print(sentence_list[i].replace(' ', ''))
        print('---------------------')

if __name__ == '__main__':
    main()
