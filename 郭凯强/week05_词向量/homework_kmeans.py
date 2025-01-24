#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
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

# 加载和预处理文本数据
def load_sentence(path):
    """
    读取文本文件，对每行文本进行分词
    返回: 分词后的句子集合，用空格连接词语
    例如："我爱北京" -> "我 爱 北京"
    """
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    """
    将句子转换为向量，方法是：
    1. 获取句子中每个词的词向量
    2. 将所有词向量求平均，作为句子向量
    """
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
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量（经验值，目的是让每个类别的大小适中）
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起

    #计算类内距离
    density_dict = defaultdict(list)
    for vector_index, label in enumerate(kmeans.labels_):
        vector = vectors[vector_index]   #某句话的向量
        center = kmeans.cluster_centers_[label]  #对应的类别中心向量
        distance = cosine_distance(vector, center)  #计算距离
        density_dict[label].append(distance)    #保存下来
    for label, distance_list in density_dict.items():
        density_dict[label] = np.mean(distance_list)   #对于每一类，将类内所有文本到中心的向量余弦值取平均
    density_order = sorted(density_dict.items(), key=lambda x:x[1], reverse=True)  #按照平均距离排序，向量夹角余弦值越接近1，距离越小

    #按照余弦距离顺序输出
    for label, distance_avg in density_order:
        print("cluster %s , avg distance %f: " % (label, distance_avg))
        sentences = sentence_label_dict[label]
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

#向量余弦距离
def cosine_distance(vec1, vec2):
    vec1 = vec1 / np.sqrt(np.sum(np.square(vec1)))  #A/|A|
    vec2 = vec2 / np.sqrt(np.sum(np.square(vec2)))  #B/|B|
    return np.sum(vec1 * vec2)

if __name__ == "__main__":
    main()