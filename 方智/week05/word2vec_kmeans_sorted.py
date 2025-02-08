#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法,按照从小到大顺序排序类内距离,选择前5个聚类,打印每个聚类的句子
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
def load_word2vec_model(path):#加载词向量模型
    model = Word2Vec.load(path)#加载模型
    return model

def load_sentence(path):#加载所有句子
    sentences = set()#句子集合
    with open(path, encoding="utf8") as f:#打开文件
        for line in f:#读取每行
            sentence = line.strip()#去掉空格
            sentences.add(" ".join(jieba.cut(sentence)))#分词并加入集合
    print("获取句子数量：", len(sentences))#打印句子数量
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):#将句子转换为向量
    vectors = []#句子向量集合
    for sentence in sentences:#遍历每句话
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)#初始化句子向量
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]#取出词向量
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))#求平均，作为句子向量
    return np.array(vectors)
#类平均距离排序



def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量,sqrt(句子数量)
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)#句子标签字典
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    # 打印聚类平均距离小的前3个
    for label, sentences in sorted(sentence_label_dict.items(), key=lambda x: kmeans.inertia_ / len(x[1]))[:3]:#打印聚类平均距离小的前3个
        print("cluster %s :" % label)#打印聚类编号
        # for sentence in sentences:  # 打印每个聚类中的句子
        #      print(sentence.replace(" ", ""))
        # print('-----------')
        print("每个类内句子数量：", len(sentences))  # 打印每个类内句子数量
        print("每个类内平均距离：", kmeans.inertia_ / len(sentences))  # 计算每个类内平均距离

if __name__ == "__main__":
    main()


