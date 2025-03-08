# -*- coding: utf-8 -*-
import csv
import json
import os

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from config import Config
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '差评', 1: '好评'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])  # Bert分词器
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    # 构造title和tag称为Tensor对象
    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as file:
            reader = csv.reader(file)
            data = list(reader)[1:]
            for line in data:
                label = int(line[0])
                title = line[1]
                if self.config["model_type"] == "bert":
                    # 如果是bert，就用bert自带的词表
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(title)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        return

    # 对一个句子encode，规则是根据词表的下标
    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    # 实现这个方法，可以使用len(obj)获取数据集的大小
    def __len__(self):
        return len(self.data)

    # 实现这个方法，可以使用索引访问类的实例，就像访问列表或字典那样,obj[1]就可以访问第二个样本
    def __getitem__(self, index):
        return self.data[index]

# 加载词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    # 将样本分为训练样本和验证样本
    # if os.path.exists('../../data/train.csv') and os.path.exists('../../data/valid.csv'):
    #     os.remove('../../data/train.csv')
    #     os.remove('../../data/valid.csv')
    # with open('../../data/total.csv', 'r') as file:
    #     reader = csv.reader(file)
    #     data = list(reader)[1:]
    # data_0 = [row for row in data if row[0] == '0']
    # data_1 = [row for row in data if row[0] == '1']
    # train_data_0 = data_0[:int(len(data_0) * 0.8)]
    # valid_data_0 = data_0[int(len(data_0) * 0.8):]
    # train_data_1 = data_1[:int(len(data_1) * 0.8)]
    # valid_data_1 = data_1[int(len(data_1) * 0.8):]
    # train_data = train_data_0 + train_data_1
    # valid_data = valid_data_0 + valid_data_1
    # with open('../../data/train.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(train_data)
    # with open('../../data/valid.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(valid_data)

    dg = DataGenerator("../../data/total.csv", Config)
    print(dg[1])
