import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

""" 
数据加载 
"""


class DataGenerator:
    def __init__(self, data_path, config):

        self.config = config  # 数据集路径
        self.path = data_path
        self.index_to_label = {0: '差评', 1: '好评'}  # 索引到标签的映射
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())  # 标签到索引的映射
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":  # 使用bert时加载tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])  # 加载预训练模型的tokenizer
        self.vocab = load_vocab(config["vocab_path"])  # 加载词表
        self.config["vocab_size"] = len(self.vocab)  # 词表大小
        self.load()


def load(self):
    self.data = []
    with open(self.path, encoding="utf8") as f:
        for line in f:
            line = json.loads(line)  # 加载数据集
            index = line["label"]  # 标签索引
            review = line["review"]  # 评论内容
        if self.config["model_type"] == "bert":
            input_id = self.tokenizer.encode(review, max_length=self.config["max_length"], pad_to_max_length=True)
        else:
            input_id = self.encode_sentence(review)
        input_id = torch.LongTensor(input_id)
        label_index = torch.LongTensor([index])
        self.data.append([input_id, label_index])
        return


def encode_sentence(self, text):
        input_id = []
    for char in text:
        input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
    input_id = self.padding(input_id)
    return input_id


# 补齐或截断输入的序列，使其可以在一个batch内运算
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
    for index, line in enumin enumerate(f): 
        token = line.strip() 
        token_dict[token] = index + 1 #0留给padding位置，所以从1开始 
    return token_dict 
  
   
#用torch自带的DataLoader类封装数据 
def load_data(data_path, config, shuffle=True): 
   dg = DataGenerator(data_path, config) 
   dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle) 
   return dl 
  
if __name__ == "__main__": 
   from config import Config 
    dg = DataGenerator("week7/work/data/test.json", Config) 
    print(dg[1]) 
