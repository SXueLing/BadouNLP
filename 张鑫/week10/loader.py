# -*- coding: utf-8 -*-

"""
数据加载
"""
import json
import random

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer


class DataGenerator:
    def __init__(self, tokenizer, data_path, config):
        self.schema = None
        self.data = None
        self.config = config
        self.path = data_path
        self.simple_size = config['simple_size']
        self.train_window_size = config['train_window_size']
        self.tokenizer = tokenizer
        self.sentences = []
        self.corpus = self.load_corpus()

    # 加载语料
    def load_corpus(self):
        corpus = ""
        with open(self.path, encoding="gbk") as f:
            for line in f:
                corpus += line.strip()
        return corpus

    def __len__(self):
        return self.simple_size

    def __getitem__(self, index):
        start = random.randint(0, len(self.corpus) - self.train_window_size - 1)
        end = start + self.train_window_size
        data = self.corpus[start:end]
        target = self.corpus[start + 1:end + 1]  # 输入输出错开一位
        # print(window, target)
        x = self.tokenizer.encode(data,
                                  add_special_tokens=False,
                                  padding="max_length",
                                  max_length=self.config["max_length"],
                                  truncation=True)
        y = self.tokenizer.encode(target,
                                  add_special_tokens=False,
                                  padding="max_length",
                                  max_length=self.config["max_length"],
                                  truncation=True)
        return torch.LongTensor(x), torch.LongTensor(y)


# 用torch自带的DataLoader类封装数据
def load_data(data_path, tokenizer, config, shuffle=True):
    dg = DataGenerator(tokenizer, data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
