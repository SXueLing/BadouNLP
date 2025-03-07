# -*- coding: utf-8 -*-

import json
import re
import os
from collections import defaultdict

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
        self.tokenizer = load_vocab(config["bert_path"])
        #self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
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
                input_ids = self.encode_sentence(sentenece,self.config["max_length"])
                labels = self.padding(labels,self.config["max_length"], -1)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    def encode_sentence(self, text, max_len ):
        input_id = self.tokenizer.encode(text,padding="max_length",max_length=max_len,truncation=True)
        return input_id


    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, max_len,pad_token=0):
        input_id = input_id[:max_len]
        input_id += [pad_token] * (max_len- len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

#加载字表或词表
def load_vocab(vocab_path):
    return BertTokenizer.from_pretrained(vocab_path)
    #token_dict = {}
    #with open(vocab_path, encoding="utf8") as f:
    #    for index, line in enumerate(f):
    #        token = line.strip()
    #        token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    #return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)

