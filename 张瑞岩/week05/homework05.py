
#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
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

def cosine_similarity(vector1, vector2):
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    norm_vector1 = math.sqrt(sum(a ** 2 for a in vector1))
    norm_vector2 = math.sqrt(sum(b ** 2 for b in vector2))
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化
    print(vectors.shape)


    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起

    # print(kmeans.cluster_centers_)

    #计算每一类中的距离和
    class_distance = [0]*n_clusters
    for vector, label in zip(vectors, kmeans.labels_):
        distance = cosine_similarity(vector, kmeans.cluster_centers_[label])
        class_distance[label] += distance
    # print(class_distance)

    #计算平均距离并与标签绑定
    class_distance_dict = defaultdict(list)
    for label, sentences in sentence_label_dict.items():
        class_distance_dict[label].append(class_distance[label]/len(sentences))
    # print(class_distance_dict)

    #按照平均距离从小到大排序
    sorted_dict = dict(sorted(class_distance_dict.items(), key=lambda item: item[1]))
    print(sorted_dict)

    sorted_dict.update(sentence_label_dict)
    for label, sentences in sorted_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()

