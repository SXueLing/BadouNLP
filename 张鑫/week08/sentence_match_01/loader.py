"""
数据加载
"""
import json
from collections import defaultdict
import random
import jieba
import torch
from torch.utils.data import DataLoader


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config['vocab_path'])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config['schema_path'])
        # 随机取样，得限制个总长度，要不一直取
        self.train_data_size = config['epoch_data_size']
        # "train" or "test"，根据传入文件单行的格式来判断
        self.data = []
        self.knwb = defaultdict(list)
        self.data_type = None
        self.load()

    def load(self):
        with open(self.path, encoding='utf8') as f:
            for line in f:
                line = json.loads(line)
                if isinstance(line, dict):
                    # 训练集
                    self.data_type = "train"
                    label = line['target']
                    questions = line['questions']
                    for question in questions:
                        input_id = torch.LongTensor(self.encode_sentence(question))
                        self.knwb[self.schema[label]].append(input_id)
                else:
                    # 测试集
                    self.data_type = "test"
                    question, label = line
                    input_id = torch.LongTensor(self.encode_sentence(question))
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])
        return

    def encode_sentence(self, sentence):
        input_id = []
        if self.config['vocab_path'] == "words.txt":
            for word in jieba.cut(sentence):
                input_id.append(self.vocab.get(word, self.vocab['[UNK]']))
        else:
            for char in sentence:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.triplet_train_sample()  # 随机生成一个训练样本
        else:
            return self.data[index]

    # 生成triplet loss对应的采样数据方法
    def triplet_train_sample(self):
        standard_question_index = list(self.knwb.keys())
        ap, n = random.sample(standard_question_index, 2)
        if len(self.knwb[ap]) < 2:
            return self.triplet_train_sample()
        a, p = random.sample(self.knwb[ap], 2)
        n = random.choice(self.knwb[n])
        return [a, p, n]


# 加载字表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


# 加载schema
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
