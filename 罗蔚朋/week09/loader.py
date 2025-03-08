# -*- coding: utf-8 -*-

"""
数据加载
"""
import json

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer


class DataGenerator:
    def __init__(self, data_path, config):
        self.schema = None
        self.data = None
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_path'])
        self.sentences = []
        self.load_schema()
        self.load()

    def load_schema(self):
        with open(self.config["schema_path"], encoding='utf8') as f:
            self.schema = json.load(f)
        return

    def load(self):
        self.data = []
        with open(self.path, encoding='utf-8') as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = [8]  # CLS token，映射成schema[8]~O
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentence))
                input_ids = self.tokenizer.encode(sentence, padding="max_length",
                                                  max_length=self.config["max_length"],
                                                  truncation=True)
                labels = self.padding(labels, -1)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
