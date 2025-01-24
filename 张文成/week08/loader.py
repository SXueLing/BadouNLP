# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        self.train_data_size = config["epoch_data_size"]
        self.data_type = None
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_train_sample()
        else:
            return self.data[index]

    def random_train_sample(self):
        # 获取所有有至少两个样本的标准问题
        valid_anchors = [p for p, samples in self.knwb.items() if len(samples) >= 2]
        anchor_p = random.choice(valid_anchors)
        # 随机选择两个不同的索引
        idx1, idx2 = random.sample(range(len(self.knwb[anchor_p])), 2)
        anchor = self.knwb[anchor_p][idx1]
        positive = self.knwb[anchor_p][idx2]
        # 选择一个不同的标准问题
        negative_p = random.choice([p for p in self.knwb.keys() if p != anchor_p])
        negative = random.choice(self.knwb[negative_p])
        return [anchor, positive, negative]



#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

#加载schema
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

