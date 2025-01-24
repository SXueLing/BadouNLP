# -*- coding: utf-8 -*-

import csv
import json
import re
import os
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config,train=True):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '差评', 1: '好评'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load(train)


    def load(self,train):
        self.data = []
        data_t = []
        data_f = []
        with open(self.path, encoding="utf8",mode='r') as f:
            csv_reader = csv.reader(f)
            max_length = defaultdict(int)
            _ = next(csv_reader)
            for row in csv_reader:
                if len(row[1]) in max_length:
                    max_length[len(row[1])] += 1
                else:
                    max_length[len(row[1])] = 1
                if row[0] == '1':
                    data_t.append(row[1])
                else:
                    data_f.append(row[1])
            sorted_max_length = sorted(max_length.items(),key=lambda x:x[1],reverse=True)
            print(sorted_max_length)

            if train:
                data_t = data_t[:int(len(data_t)*0.8)]
                data_f = data_f[:int(len(data_t)*0.8)]
            else:
                data_t = data_t[int(len(data_t) * 0.8):]
                data_f = data_f[int(len(data_t) * 0.8):]
            for value in data_t:
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(value, max_length=self.config["max_length"],
                                                     pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(value)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([1])
                self.data.append([input_id, label_index])
            for value in data_f:
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(value, max_length=self.config["max_length"],
                                                     pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(value)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([0])
                self.data.append([input_id, label_index])

        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True ,train=True):
    dg = DataGenerator(data_path, config,train)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("data.csv", Config)
    print(dg[1])
