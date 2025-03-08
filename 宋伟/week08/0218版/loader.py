'''
1. 数据预处理
2. dataset 与 dataloaer 的实现

[description]
'''

import json
import torch
import random
import jieba
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


class MyDataSet(Dataset):
    """docstring for MyDataSet"""

    def __init__(self, data_path, config):
        super(MyDataSet, self).__init__()
        self.path = data_path  # 两种数据来源，训练集或
        self.config = config
        self.vocab = load_vocab(config["vocab_path"])  # 加载字表{word:index}
        self.schema = load_schema(config["schema_path"])  # 加载标准问-索引 {standard_question:index}
        self.train_data_size = config["epoch_data_size"]
        self.data_type = None  # 表示数据是训练集还是测试集
        self.loss = config["loss"]
        self.config["vocab_size"] = len(self.vocab)
        self.load()

    def load(self):
        self.data = []  # 用于记录测试集中的数据[[token_ids,label_index]]
        # knowledge base用于记录训练集每个{label_index:[token_ids]}
        self.knwb = defaultdict(list)
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        token_ids = self.encode_sentence(question)
                        token_ids = torch.LongTensor(token_ids)
                        self.knwb[self.schema[label]].append(token_ids)
                        # 这里貌似没有对schema[label]进行长整化
                        # 因为这里仅仅只是，使用schema[label],来抽样获取正负样本对
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    token_ids = self.encode_sentence(question)
                    token_ids = torch.LongTensor(token_ids)
                    label_index = torch.LongTensor(
                        [self.schema[label]])  # 标量进行向量化
                    self.data.append([token_ids, label_index])    
        return

    def encode_sentence(self, text):
        token_ids = []
        # 如果按词进行分词
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                token_ids.append(self.vocab.get(word, self.vocab["[UNK]"]))
        # 如果按字进行分词
        else:
            for char in text:
                token_ids.append(self.vocab.get(char,self.vocab["[UNK]"]))

        # 进行索引码的规整化，补全或截断
        token_ids = self.padding(token_ids)
        return token_ids

    def padding(self, token_ids):
        token_ids = token_ids[:self.config["max_length"]]
        token_ids += [0]*(self.config["max_length"]-len(token_ids))
        return token_ids

    def __len__(self):
        # 如果是训练集，则训练集的大小，保持在`rpoch_data_size`
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        # 如果是测试集，直接返回测试集大小即可
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    def __getitem__(self, idx):
        # 需要主要训练使用的数据格式与验证使用的数据格式是不同的
        # 如果是训练集，随机生成一个训练样本
        if self.loss=="cos_loss":
            if self.data_type == "train":
                return self.random_train_sample(self.loss)  # 数据对是 [s1,s2,match]
            # 直接提取一条样本即可 （token_ids,label_index）
            else:
                return self.data[idx]  # 数据对是 （token_ids,label_index）
        elif self.loss == "triplet_loss":
            if self.data_type == "train":
                return self.random_train_sample(self.loss)  # s数据对是[a,p,n]


    def random_train_sample(self,loss="cos_loss"):
        standard_question_index = list(self.knwb.keys())  # 获取标准问索引
        if loss == "cos_loss":
        # 随机生成一个正样本
            if random.random() <= self.config["positive_sample_rate"]:
                p = random.choice(standard_question_index)
                if len(self.knwb[p]) < 2:
                    return self.random_train_sample(loss)  # 重新进行选择
                else:
                    s1, s2 = random.sample(self.knwb[p], 2)
                    return [s1, s2, torch.LongTensor([1])]  # 通过1，标注为正样本
            else:
                p, n = random.sample(standard_question_index, 2)
                s1 = random.choice(self.knwb[p])
                s2 = random.choice(self.knwb[n])
                return [s1, s2, torch.LongTensor([-1])]  # 通过-1 标注为负样本
        elif loss == "triplet_loss":
            a,n = random.sample(standard_question_index,2)
            if len(self.knwb[a]) <2:
                return self.random_train_sample(loss)
            else:
                anchor_vector,positive_vector = random.sample(self.knwb[a],2)
                negative_vector = random.choice(self.knwb[n])
                return [anchor_vector,positive_vector,negative_vector]

    # 加载字表或词表


def load_vocab(vocab_path):
    # tokrn-index
    token_dict = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0是预留给padding
    return token_dict


def load_schema(schema_path):
    # 该文件是一个标准的json 数据格式，将其转化为python 数据格式
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.loads(f.read())


def load_data(data_path, config, shuffle=True):
    dataset = MyDataSet(data_path, config)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=config["batch_size"],
                            shuffle=shuffle)
    return dataloader


if __name__ == '__main__':
    from config import Config
    Config["loss"] = "triplet_loss"
    dataloader = load_data(Config["train_data_path"],Config)
    # 典型的小批量的随机梯度下降策略
    for epoch in range(10):
        for i,batch_data in enumerate(dataloader):   # 每轮训练，随机从中取batch

            s1,s2,relative = batch_data
            print(s1.shape)
            print(relative[-1])
        print(f'{"separation":=^50}')
        


