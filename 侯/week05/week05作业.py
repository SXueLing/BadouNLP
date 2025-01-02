"""
@Project ：cgNLPproject
@File    ：week05_02.py
@Date    ：2024/12/26 12:22
使用kmeans，将一个文本进行聚集分类
第一步：
训练出词向量
第二步：
根据词向量，得到目标文本，每句话的句向量（此处使用的是将所有词向量加起来取平均）
第三步：
使用kmeans，对所有句向量进行分类
第四步：
对分类好的所有类别，做类内平均距离计算统计
第五步：
根据距离排序
"""
"""
@Project ：cgNLPproject
@File    ：week05_02.py
@Date    ：2024/12/26 12:22
使用kmeans，将一个文本进行聚集分类
第一步：
训练出词向量
第二步：
根据词向量，得到目标文本，每句话的句向量（此处使用的是将所有词向量加起来取平均）
第三步：
使用kmeans，对所有句向量进行分类
第四步：
对分类好的所有类别，做类内平均距离计算统计
第五步：
根据距离排序
"""
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import math
from collections import defaultdict

def load_w2v_module(path):
    w2v = Word2Vec.load(path)
    return w2v

def sentence_transform_vector(w2vModule_, sentences_):
    sent2vc = []
    for sentence in sentences_:
        sentenv = np.zeros(w2vModule_.vector_size)
        for word_ in sentence:
            try:
                sentenv += w2vModule_.wv[word_]
            except KeyError:
                sentenv += np.zeros(w2vModule_.vector_size)
        sentenv = sentenv / len(sentence)
        sent2vc.append(sentenv)
    return sent2vc



# 对所有句子做聚集分类
def kmeans_train(k_,sentence_):
    kmeans_module_ = KMeans(k_)
    kmeans_module_.fit(sentence_)
    return kmeans_module_

def main():
    w2v = load_w2v_module('model.w2v')
    sentences = set()
    with open('titles.txt', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            sentences.add(" ".join(jieba.lcut(line)))
    print('句子数量：', len(sentences))
    # 将句子转换成向量
    sentences = list(sentences)
    s2v = sentence_transform_vector(w2v, sentences)
    kmeans_center_num = int(math.sqrt(len(sentences)))
    kmeans = kmeans_train(kmeans_center_num, s2v)
    # 存储欧几里得距离
    # （key:label, value:{key:sentence_index, value:norm}）
    k_norm = defaultdict(dict)
    # k_norm = defaultdict(list)
    # 句子索引
    sentence_index = 0
    for sentence_v, label_num in zip(s2v, kmeans.labels_):
        center_vc = kmeans.cluster_centers_[label_num]
        # numpy库的
        dis_norm = abs(np.linalg.norm(sentence_v - center_vc))
        if label_num not in k_norm:
            k_norm[label_num] = {}
        if sentence_index not in k_norm[label_num]:
            k_norm[label_num][sentence_index] = {}
        k_norm[label_num][sentence_index] = dis_norm
        sentence_index += 1
    label_norm = {}
    sentence_sorted = {}
    for label, sentence_norm in k_norm.items():
        # print(f' k_norm[{label}]:', k_norm[label])
        # print('label:', label)
        if label not in label_norm:
            label_norm[label] = {}
            label_norm[label] = sum(sentence_norm.values()) / len(sentence_norm)
        k_norm[label] = dict(sorted(sentence_norm.items(), key=lambda x: x[1]))
        # print(f' k_norm[{label}]:', k_norm[label])
    label_sorted = sorted(label_norm.items(), key=lambda x: x[1])
    print('label_sorted:', label_sorted)
    num = 0
    for label, norm in label_sorted:
        num += 1
        print(f'第{num}个label:{label}')
        a = list(k_norm[label].keys())[:min(5, len(k_norm[label].keys()))]
        for i, s in enumerate(a):
            print(f'第{i+1}个sentence:{sentences[s].replace(' ','')}')

            # for i in range(min(len(k_norm[label]), 10)):
            # print(f'第{i+1}个sentence:{}')

if __name__ == '__main__':
    main()
