import math
import jieba
import re
import os
import json
from collections import defaultdict
import pandas as pd

jieba.initialize()

"""
贝叶斯分类实践

P(A|B) = (P(A) * P(B|A)) / P(B)
事件A：文本属于类别x1。文本属于类别x的概率，记做P(x1)
事件B：文本为s (s=w1w2w3..wn)
P(x1|s) = 文本为s，属于x1类的概率.   #求解目标#
P(x1|s) = P(x1|w1, w2, w3...wn) = P(w1, w2..wn|x1) * P(x1) / P(w1, w2, w3...wn)

P(x1) 任意样本属于x1的概率。x1样本数/总样本数
P(w1, w2..wn|x1) = P(w1|x1) * P(w2|x1)...P(wn|x1)  词的独立性假设
P(w1|x1) x1类样本中，w1出现的频率

公共分母的计算，使用全概率公式：
P(w1, w2, w3...wn) = P(w1,w2..Wn|x1)*P(x1) + P(w1,w2..Wn|x2)*P(x2) ... P(w1,w2..Wn|xn)*P(xn)
"""


class BayesApproach:
    def __init__(self, data_path):
        self.p_class = defaultdict(int)  # 每个类别的文本数量
        self.word_class_prob = defaultdict(dict)
        self.load(data_path)

    def load(self, path):
        """
        1. 初始化变量  类别的词频、词表
        2. 读取tag和title
        3. 将tag设置为类别，title分词，记录词频
        4. 转换为概率
        :param path:
        :return:
        """
        self.class_name_to_frequcy = defaultdict(dict)
        self.all_words = set()
                # 读取 CSV 文件
        df = pd.read_csv(path)
        
        # 遍历数据
        for _, row in df.iterrows():
            class_name = row["label"]  # 类别是 label
            title = row["review"]  # 评论是 review
            
            # 分词
            words = jieba.lcut(title)
            
            # 更新类别频次
            self.p_class[class_name] += 1
            
            # 更新所有词集合
            self.all_words = self.all_words.union(set(words))
            
            # 更新词频字典
            word_freq = self.class_name_to_frequcy[class_name]
            for word in words:
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
        self.freq_to_prob()
        return

    # 将记录的词频和样本频率都转化为概率
    def freq_to_prob(self):
        # 样本概率
        total_sample_num = sum(self.p_class.values())
        self.p_class = dict([class_name, self.p_class[class_name] / total_sample_num] for class_name in self.p_class)
        # 词概率
        self.word_prob = defaultdict(dict)
        for class_name, word_freq in self.class_name_to_frequcy.items():
            total_word_of_class = sum(count for count in word_freq.values())  # 重复的词要记录全部次数
            for word in word_freq:
                current_word_prob = (word_freq[word] + 1) / (total_word_of_class + len(self.all_words))
                self.word_prob[class_name][word] = current_word_prob
            self.word_prob[class_name]["<unk>"] = 1 / (total_word_of_class + len(self.all_words))
        return

    # P(w1|x1) * P(w2|x1)...P(wn|x1)
    def get_words_class_prob(self, words, class_name):
        result = 1
        for word in words:
            unk_prob = self.word_prob[class_name]["<unk>"]
            result *= self.word_prob[class_name].get(word, unk_prob)
        return result

    # 计算P(w1, w2..wn|x1) * P(x1)
    def get_class_prob(self, words, class_name):
        p_class = self.p_class[class_name]
        p_class_w = self.get_words_class_prob(words, class_name)
        return p_class * p_class_w

    # 做文本分类
    def classify(self, sentence):
        words = jieba.lcut(sentence)
        probs = []
        for tag in self.p_class.keys():
            probs.append([tag,self.get_class_prob(words,tag)])

        probs = sorted(probs, key=lambda x: x[1], reverse=True)
        # 计算公共分母：P(w1, w2, w3...wn) = P(w1,w2..Wn|x1)*P(x1) + P(w1,w2..Wn|x2)*P(x2) ... P(w1,w2..Wn|xn)*P(xn)
        # 不做这一步也可以，对顺序没影响，只不过得到的不是0-1之间的概率值
        pw = sum([x[1] for x in probs])  # P(w1, w2, w3...wn)
        results = [[c, prob / pw] for c, prob in probs]

        for class_name,prob in results:
            print("属于类别[%s]的概率为%f" % (class_name, prob))
        return results

if __name__ == "__main__":
    path = "./data.csv"
    ba = BayesApproach(path)
    query = "送过来菜还是热的，很香"
    ba.classify(query)
