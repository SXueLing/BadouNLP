#conding :utf-8
'''
基于训练好的词向量，进行Kmeans聚类算法
'''
import math
import re
import json

import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#加载训练好的模型
def load_model(path):
    model = Word2Vec.load(path)
    return model

# 获取分词结果
def load_sentence(path):
    sentences = set()
    with open(path,encoding='utf8') as f:
        for line in f:
            sentence = line.strip()
            sentences.add(''.join(jieba.cut(sentence)))
    print('获取句子数量：' ,len(sentences))
    return sentences

# 文本向量化
def sentence_to_vec(sentences,model):
    vecs = []
    for sentence in sentences:
        words = sentence.split()
        vec = np.zeros(model.vector_size)
        # 所有向量加和求平均，作为句子向量
        for word in words:
            try:
                vec += model.mv[word]
            except KeyError:
                vec += np.zeros(model.vector_size)
        vecs.append(vec/len(words))
    return np.array(vecs)

def cosine_distance(vec1, vec2):
    vec1 = vec1 / np.sqrt(np.sum(np.square(vec1)))  #A/|A|
    vec2 = vec2 / np.sqrt(np.sum(np.square(vec2)))  #B/|B|
    return np.sum(vec1 * vec2)

def eculid_distance(vec1,vec2):
    return np.sqrt((np,sum(np.square(vec1 - vec2))))

def main():
    model = load_model('model.w2v')
    sentences = load_sentence('titles.txt')
    vecs = sentence_to_vec(sentences,model)
    n_clusters = int(math.sqrt(len(sentences)))
    print('指定聚类数量',n_clusters)
    kmeans = KMeans(n_clusters)
    kmeans.fit(vecs)

    sentence_label_dict = defaultdict(list)
    for sentence , lable  in zip(sentences,kmeans.labels_):
        sentence_label_dict[lable].append(sentence)


    #计算类内距离
    density_dict = defaultdict(list)
    for vec_index ,label in enumerate(kmeans.labels_):
        vec = vecs[vec_index]
        center = kmeans.cluster_centers_[label]
        distance = cosine_distance(vec,center)
        density_dict[label].append(distance)
    for label,distance_list in density_dict.items():
        density_dict[label] = np.mean(distance_list)

    density_order = sorted(density_dict.items(),key=lambda x:x[1], reverse=True)

    # 按照余弦距离顺序输出
    for label,distance_avg in density_order:
        print('cluster %s , avg distance %f:' % (label,distance_avg))
        sentences = sentence_label_dict[label]
        for i in range(min(10,len(sentences))):
            print(sentences[i].replace(' ',''))
        print('****************')

if __name__ == '__main__':
    main()
