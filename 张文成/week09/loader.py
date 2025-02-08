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
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.config["vocab_size"] = self.tokenizer.vocab_size
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
                input_ids = self.encode_sentence(sentenece)
                # 处理标签：添加-100并调整长度
                labels = labels[:self.config["max_length"]-2]  # 截断到最大长度-2
                labels = [-100] + labels + [-100]  # 添加特殊标记位置
                labels = self.padding(labels, -100)  # 填充到max_length
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    def encode_sentence(self, text):
        """使用BERT tokenizer编码文本，自动添加特殊标记"""
        text = "".join(text)
        inputs = self.tokenizer.encode_plus(
            text,
            max_length=self.config["max_length"],
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            return_tensors=None
        )
        return inputs["input_ids"]

    def padding(self, input_id, pad_token=0):
        """将序列补齐或截断到max_length"""
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

# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)
