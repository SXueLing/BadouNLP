# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config, logger):
        self.config = config
        self.logger = logger
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.config["pad_idx"] = self.vocab["[PAD]"]
        self.config["start_idx"] = self.vocab["[CLS]"]
        self.config["end_idx"] = self.vocab["[SEP]"]
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                title = line["title"]
                content = line["content"]
                self.prepare_data(title, content)

    #文本到对应的index
    #头尾分别加入[cls]和[sep]
    def encode_sentence(
        self,
        question,
        answer,
    ):
        # 之前的那种分别输入输出编码的不太适合，重新写成把输入输出一次性编码的方式
        raw_input_ids = [
            self.vocab.get(char, self.vocab["[UNK]"]) for char in question
        ]
        raw_output_ids = [
            self.vocab.get(char, self.vocab["[UNK]"]) for char in answer
        ]
        input_id = raw_input_ids + [self.vocab["[SEP]"]] + raw_output_ids
        raw_output_ids.append(self.vocab["[SEP]"])
        output_id = [0] * len(raw_input_ids) + raw_output_ids # label前面补0
        input_id = self.padding(input_id, self.config["input_max_length"])
        output_id = self.padding(output_id, self.config["output_max_length"])
        return input_id, output_id


    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, length):
        input_id = input_id[:length]
        input_id += [self.vocab["[PAD]"]] * (length - len(input_id))
        return input_id

    #输入输出转化成序列
    def prepare_data(self, title, content):
        input_seq, output_seq = self.encode_sentence(title, content)
        self.data.append([
            torch.LongTensor(input_seq),
            torch.LongTensor(output_seq)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index
    return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, logger, shuffle=True):
    dg = DataGenerator(data_path, config, logger)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dl = load_data(Config["train_data_path"], Config, 1)
    print(dl[1])

