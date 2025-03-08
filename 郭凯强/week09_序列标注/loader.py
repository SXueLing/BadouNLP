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
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentence))

                # BERT编码
                encoded = self.encode_sentence(sentence)
                input_ids = encoded['input_ids']
                attention_mask = encoded['attention_mask']
                labels = self.padding(labels, -100)  # BERT常用-100作为标签的padding值

                self.data.append([torch.LongTensor(input_ids),  torch.LongTensor(attention_mask), torch.LongTensor(labels)])
        return

    def encode_sentence(self, text):
        # 使用BERT tokenizer进行编码
        tokens = ['[CLS]']  # 添加起始标记
        for char in text:
            tokens.append(char)
        tokens.append('[SEP]')  # 添加结束标记

        # 转换为ID
        input_ids = []
        for token in tokens:
            input_ids.append(self.tokenizer.convert_tokens_to_ids(token))

        # Padding
        attention_mask = [1] * len(input_ids)
        padding_length = self.config["max_length"] - len(input_ids)

        if padding_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
        else:
            input_ids = input_ids[:self.config["max_length"]]
            attention_mask = attention_mask[:self.config["max_length"]]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

    def padding(self, input_id, pad_token=-100):
        # 考虑CLS和SEP标记的位置
        if len(input_id) > self.config["max_length"] - 2:  # 减2是因为要为CLS和SEP留位置
            input_id = input_id[:self.config["max_length"] - 2]

        # 为CLS标记添加padding标签
        padded_labels = [pad_token]
        padded_labels.extend(input_id)
        # 为SEP标记添加padding标签
        padded_labels.append(pad_token)

        # 补充padding
        padding_length = self.config["max_length"] - len(padded_labels)
        if padding_length > 0:
            padded_labels += [pad_token] * padding_length

        return padded_labels

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
    dg = DataGenerator("../ner_data/train.txt", Config)

