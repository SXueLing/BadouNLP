# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import numpy as np
from torch import parse_ir
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        # 使用 BERT 分词器
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_model_path"])
        self.sentences = []
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
                input_ids, attention_mask, labels = self.encode_sentence(sentence, labels)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.LongTensor(labels)])
        return

    def encode_sentence(self, text, labels):
        # 添加 [CLS] 标记
        input_ids = [self.tokenizer.cls_token_id]
        new_labels = [-1]  # [CLS] 对应的标签设为 -1
        for char, label in zip(text, labels):
            # 对每个字符进行分词
            tokens = self.tokenizer.tokenize(char)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids.extend(token_ids)
            # 对于多字分词，只保留第一个字的标签，其余设为 -1
            new_labels.append(label)
            new_labels.extend([-1] * (len(token_ids) - 1))
        # 添加 [SEP] 标记
        input_ids.append(self.tokenizer.sep_token_id)
        new_labels.append(-1)

        # 补齐或截断
        input_ids = self.padding(input_ids)
        new_labels = self.padding(new_labels, -1)
        # 生成注意力掩码
        attention_mask = [1 if i != 0 else 0 for i in input_ids]

        return input_ids, attention_mask, new_labels

    # 补齐或截断输入的序列，使其可以在一个 batch 内运算
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


# 用 torch 自带的 DataLoader 类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config
    train_data = load_data(Config["train_data_path"], Config)
    for index, batch_data in enumerate(train_data):
        input_id, labels = batch_data
