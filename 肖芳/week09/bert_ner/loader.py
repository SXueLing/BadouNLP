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
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.tokenizer = BertTokenizer.from_pretrained("/Users/fancy/workspaces/NLP/bert-base-chinese")
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentenece = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentenece.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentenece))

                input_ids = self.tokenizer.encode(sentenece, max_length=100, pad_to_max_length=True)
                # print("tokens:", self.tokenizer.convert_ids_to_tokens(input_ids))  # 显示每个位置对应的token
                labels = self.padding(labels, -1)
                # print("labels.len", len(labels))
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return
    
    # 在第一位添加一个padding作为cls对应的分类
    def padding(self, input_id, pad_token=0):
        input_id = [-1] + input_id
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

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
    dg = DataGenerator("ner_data/train", Config)
    print("x", dg[0][0])
    print("y", dg[0][1])
    # labels = [8,8,8]
    # print(padding(labels, -1))
