# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_raw, config):
        self.config = config
        self.data_raw = data_raw
        self.index_to_label = {0: 0, 1: 1}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    def load(self):
        self.data = []
        for line in self.data_raw:
            tag = line["label"]
            label = self.label_to_index[tag]
            title = line["review"]
            if self.config["model_type"] == "bert":
                input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], pad_to_max_length=True)
            else:
                input_id = self.encode_sentence(title)
            input_id = torch.LongTensor(input_id)
            label_index = torch.LongTensor([label])
            self.data.append([input_id, label_index])
            # print(self.data)
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

def data_split(data_path):
    train_set = []
    test_set = []
    data = pd.read_csv(data_path)
    x = data['review']
    y = data['label']
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # 将训练集和测试集的数据配对并添加到对应的列表中
    for review, label in zip(X_train, Y_train):
        train_set.append({'review': review, 'label': label})

    for review, label in zip(X_test, Y_test):
        test_set.append({'review': review, 'label': label})
    return train_set, test_set


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    train_set, _ = data_split(data_path)
    dg = DataGenerator(train_set, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

def test_data(data_path, config, shuffle=True):
    _, test_set = data_split(data_path)
    dg = DataGenerator(test_set, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config
    raw = load_data(r"C:\py_xm\nlp01\week7\文本分类练习.csv", Config)
    print(raw)
    # load_data(Config["train_data_path"], Config)