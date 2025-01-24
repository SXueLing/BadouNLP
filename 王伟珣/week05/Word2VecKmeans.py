#coding: utf-8

import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型
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

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model("models\\model.w2v") #加载词向量模型
    sentences = load_sentence("models\\titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    # sentence_label_dict = defaultdict(list)
    # for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
    #     sentence_label_dict[label].append(sentence)         #同标签的放到一起

    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

    sentence_idx_label_dict = defaultdict(list)
    for i in range(len(kmeans.labels_)):
        sentence_idx_label_dict[kmeans.labels_[i]].append(i)

    for label in sentence_idx_label_dict.keys():
        center = kmeans.cluster_centers_[label]
        idx_array = np.array(sentence_idx_label_dict[label])
        distances = np.array([np.linalg.norm(vectors[i]-center) for i in idx_array])
        sentence_idx_label_dict[label] = idx_array[np.argsort(distances)]
    
    sentences_sorted = sorted(sentences)
    for label, idxs in sentence_idx_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(idxs))):
            print(sentences_sorted[idxs[i]].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()
