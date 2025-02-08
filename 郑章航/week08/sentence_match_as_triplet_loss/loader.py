# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import logging
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from transformers import BertTokenizer
"""
数据加载
"""

logging.getLogger("transformers").setLevel(logging.ERROR)

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.tokenizer.vocab)
        self.schema = load_schema(config["schema_path"])
        self.train_data_size = config["epoch_data_size"]  # 由于采取随机采样，所以需要设定一个采样数量，否则可以一直采
        self.max_length = config["max_length"]
        self.data_type = None  # 用来标识加载的是训练集还是测试集 "train" or "test"
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                # 加载训练集
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        self.knwb[self.schema[label]].append(question)
                # 加载测试集
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([question, label_index])
        return

    # 每次加载三个文本，输出他们的拼接后编码（anchor, positive, negative）
    def encode_sentence(self, text1, text2, text3):
        anchor_id = self.tokenizer.encode(text1,
                                          truncation='longest_first',
                                          max_length=self.max_length,
                                          padding='max_length',
                                          )
        positive_id = self.tokenizer.encode(text2,
                                            truncation='longest_first',
                                            max_length=self.max_length,
                                            padding='max_length',
                                            )
        negative_id = self.tokenizer.encode(text3,
                                            truncation='longest_first',
                                            max_length=self.max_length,
                                            padding='max_length',
                                            )
        return anchor_id, positive_id, negative_id

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_train_sample()  # 随机生成一个训练样本
        else:
            return self.data[index]

    # 依照一定概率生成负样本或正样本
    # 负样本从随机两个不同的标准问题中各随机选取一个
    # 正样本从随机一个标准问题中随机选取两个
    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())
        # 随机正样本
        p = random.choice(standard_question_index)
        # 如果选取到的标准问下不足两个问题，则无法选取，所以重新随机一次
        if len(self.knwb[p]) < 2:
            return self.random_train_sample()
        else:
            anchor, positive = random.sample(self.knwb[p], 2)
        # 随机负样本
        n = random.choice([i for i in standard_question_index if i != p])
        negative = random.choice(self.knwb[n])

        anchor_id, positive_id, negative_id = self.encode_sentence(anchor, positive, negative)
        anchor_id = torch.LongTensor(anchor_id)
        positive_id = torch.LongTensor(positive_id)
        negative_id = torch.LongTensor(negative_id)
        return anchor_id, positive_id, negative_id


# 加载字表或词表
def load_vocab(vocab_path):
    tokenizer = BertTokenizer(vocab_path)
    return tokenizer

# 加载schema
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())

# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../data/valid.json", Config)