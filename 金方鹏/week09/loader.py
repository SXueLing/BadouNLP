# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"])
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
                    sentenece.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentenece))
                #print(labels)
                input_ids, labels = self.encode_sentence(sentenece,labels)
                #print(input_ids, labels)
                labels = self.padding(labels, -1)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    def encode_sentence(self, text, labels, padding=True):
        # 使用BERT tokenizer进行编码
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.config["max_length"],
                                  return_tensors='pt', is_split_into_words=True)

        input_ids = encoding['input_ids'].squeeze(0)  # 去掉batch维度
        word_ids = encoding.word_ids()  # 获取每个 token 对应的原始字符索引
        #print(word_ids)
        # 处理标签对齐：将标签映射到子词上
        aligned_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            #print(word_idx)
            if word_idx is None:  # 特殊 token ([CLS], [SEP], [PAD])
                aligned_labels.append(self.schema['O'])  # 填充标签
            elif word_idx != previous_word_idx:  # 当前 token 是新的字符
                try:
                    aligned_labels.append(labels[word_idx])
                except KeyError:
                    aligned_labels.append(self.schema['O'])
                #aligned_labels.append(self.schema[labels[word_idx]])
            else:  # 当前 token 是字符的子词
                aligned_labels.append(self.schema['O'])  # 子词继承原词的标签
            previous_word_idx = word_idx
        #print(input_ids, aligned_labels)
        return input_ids, aligned_labels

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
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

#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("ner_data/train", Config)

