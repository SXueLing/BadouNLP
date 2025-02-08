# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import random
from config import Config
import pandas as pd
"""
数据加载
"""
random.seed(Config["seed"])

class DataGenerator:
    def __init__(self, data_path, config, is_train=True):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '差评', 1: '好评'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load(is_train)


    def load(self, is_train=True):
        # 读取 csv 文件
        self.data = []
        train_data_rate = self.config["train_data_rate"]#训练集占比
        csv_data = np.array(pd.read_csv(self.path))
        random.shuffle(csv_data)
        train_data = csv_data[:int(train_data_rate * len(csv_data))] #训练数据
        test_data = csv_data[int(train_data_rate * len(csv_data)):-1] #测试数据
        print(len(train_data), len(test_data))

        temp_data = train_data if is_train else test_data

        for [label, comment, *_] in temp_data:
            token = None
            if self.config["model_type"] == "bert":
                token = self.tokenizer.encode(comment, max_length=self.config["max_length"], pad_to_max_length=True)
            else:
                token = self.encode_sentence(comment)
            token = torch.LongTensor(token)
            label = torch.LongTensor([label])

            self.data.append([token, label])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, is_train=True, shuffle=True):
    dg = DataGenerator(data_path, config, is_train)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("文本分类练习.csv", Config)
    print(dg[1])
