# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained("/Users/fancy/workspaces/NLP/bert-base-chinese")
        self.corpus = ""
        self.load()

    def load(self):
        with open(self.path, encoding="gbk") as f:
            for line in f:
                self.corpus += line.strip()

    def __len__(self):
        return self.config["sample_len"]
    
    
    def build_sample(self, corpus):
        window_size = self.config["max_length"]
        start = random.randint(0, len(corpus) - 1 - window_size)
        end = start + window_size
        input = corpus[start:end]
        target = corpus[start + 1:end + 1]
        return input, target

    def __getitem__(self, index):
        input, target = self.build_sample(self.corpus)
        
        input_ids = self.tokenizer.encode(input, max_length=100, pad_to_max_length=True, truncation=True, add_special_tokens=False)
        labels = self.tokenizer.encode(target, max_length=100, pad_to_max_length=True, truncation=True, add_special_tokens=False)

        return [torch.LongTensor(input_ids), torch.LongTensor(labels)]

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

def padding(input_id, pad_token=0):
    input_id = [-1] + input_id
    input_id = input_id[:10]
    input_id += [pad_token] * (10 - len(input_id))
    return input_id

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("corpus.txt", Config)
    print("x", dg[0][0])
    print("y", dg[0][1])
