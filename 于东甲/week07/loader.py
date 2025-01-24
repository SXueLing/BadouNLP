# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import csv


"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '差评', 1: '好评'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    def load(self):

        with open('../文本分类练习.csv', 'r', encoding='utf-8') as file:
            # 创建CSV读取器
            reader = csv.reader(file)
            self.data =[]
            data_sum = 0
            data_right = 0
            # 迭代读取每一行
            for row in reader:
                if (row[0] == 'label'):
                    continue
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(row[1], max_length=self.config["max_length"],
                                                     padding='max_length')
                else:
                    input_id = self.encode_sentence(row[1])
                data_sum += 1
                data_right += int(row[0])
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor(int(row[0]))
                self.data.append([input_id, label_index])
            print("已构建文本分类")
            print("样本一共有：", data_sum)
            print("其中正样本有：", data_right)

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
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("valid_tag_news.json", Config)
    print(dg[2])


#只做出加载数据  理不清楚。。
