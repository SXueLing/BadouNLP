# -*- coding: utf-8 -*-

import json
# import re
# import os
import torch
# import random
# import jieba
# import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载
"""


class DataGenerator:

    def __init__(self, data_path, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(
            self.config["bert_path"])
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
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
                    sentenece.append(char)  # sentence.type: list
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentenece))
                input_ids = self.encode_sentence(sentenece)
                labels = self.padding(labels, -1)
                self.data.append(
                    [torch.LongTensor(input_ids),
                     torch.LongTensor(labels)])
                # self.data 的形状是 (num_samples, 2)，其中每个元素的形状是 (max_length,)。
        return

    # bert编码
    def encode_sentence(self, text, padding=True):
        # 使用BERT的tokenizer对文本进行分词
        tokens = self.tokenizer.tokenize("".join(text))
        # 将分词后的结果转换为BERT模型可以接受的输入格式
        input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        if padding:
            input_id = self.padding(input_id)
        return input_id  #shape: (max_length,)

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token_id=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token_id
                     ] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)


#加载字表或词表，如果用bert需要用bert的词表，预测的编码也要用bert的编码
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True, batch_size=16):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)
