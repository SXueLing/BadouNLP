# -*- coding: utf-8 -*-

import pandas as pd
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
        self.config["class_num"] = 2
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            reader = pd.read_csv(f, chunksize=1)
            for line in reader:
                label = line['label'].iloc[0]
                review = line["review"].iloc[0]
                if self.config["model_type"] == "bert":
                    # input_id = self.tokenizer.encode(review, max_length=self.config["max_length"], pad_to_max_length=True)
                    input_id = self.tokenizer.encode(review, padding = "max_length", max_length = self.config["max_length"], truncation = True)
                else:
                    input_id = self.encode_sentence(review)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
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
        """
        对输入的id序列进行padding操作，使其长度等于max_length。
        
        参数:
        input_id (list): 输入的id序列。

        返回:
        list: 经过padding后的id序列。
        """
        # 截断输入序列，使其长度不超过max_length
        input_id = input_id[:self.config["max_length"]]
        # 在输入序列的末尾添加0，使其长度等于max_length
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
    print(dg[1])
