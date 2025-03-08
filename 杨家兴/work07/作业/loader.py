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
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '差评', 1: '好评'} # 动态的保存到config里，不能写死在config
        self.label_to_index = dict((y,x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding='utf8') as f:
            for line in f:
                # print(line, 'line')
                line = line.split(",")
                tag = line[0]
                # label = self.label_to_index[tag] # 0,1,2,3
                label = int(tag) # 直接用数字
                title = line[1] # 内容
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(title)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        # 获取每个词对应的id
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id
    
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        # 因为不够长的补0，所以词表的索引是index+1,0是用来补齐长度的
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id
    
    def __len__(self):
        # print(len(self.data), 'len(self.data)')
        return len(self.data)
    
    def __getitem__(self, index):
        # print(self.data[1], 'len(self.data)len(self.data)')
        return self.data[index]

# chars.txt加载词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding='utf8') as f:
        for index, line in enumerate(f):
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
    dg = DataGenerator("train_data.txt", Config)
    print(dg[0], 'dg[0]')
    # 第一条属于娱乐，对应的index是14
    # [tensor([ 101, 1184, 5683, 3215, 2571, 7623, 2421, 2802, 2339, 3198, 5959, 8132,
    #     1039, 3295,  680, 3448, 3308,  836, 1394,  868,  102,    0,    0,    0,
    #        0,    0,    0,    0,    0,    0]), tensor([14])]
    #如果是bert的话，从bert里的词表里获取