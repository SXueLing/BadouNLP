'''
Author: Zhao
Date: 2025-01-14 20:35:21
LastEditTime: 2025-01-15 18:23:45
FilePath: loader.py
Description: 数据生成DataGenerator及其相关函数,
        用于加载数据并生成三元组样本(Anchor, Positive, Negative),以支持使用Triplet Loss的模型训练

'''

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class DataGenerator:
    def __init__(self, data_path, config):
        # 初始化DataGenerator，加载配置和数据路径
        self.config = config
        self.path = data_path
        # 加载词汇表，计算词汇表大小
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        # 加载标签schema
        self.schema = load_schema(config["schema_path"])
        # 设置训练数据大小
        self.train_data_size = config["epoch_data_size"]
        self.data_type = None
        # 加载数据
        self.load()
    
    def load(self):
        # 初始化数据和知识库
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.path, encoding='utf8') as f:
            for line in f:
                line = json.loads(line)
                if isinstance(line, dict):
                    # 如果是训练数据
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        # 编码问题
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        # 将编码后的问题添加到知识库中
                        self.knwb[self.schema[label]].append(input_id)
                else:
                    # 如果是测试数据
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    # 编码问题
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    # 将问题和标签添加到数据集中
                    self.data.append([input_id, label_index])
        return
    
    def encode_sentence(self, text):
        # 将输入的文本编码为ID序列
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            # 如果使用词汇表分词
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            # 如果使用字符表分词
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        # 填充ID序列
        input_id = self.padding(input_id)
        return input_id
    
    def padding(self, input_id):
        # 对ID序列进行填充，保证长度一致
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id
    
    def __len__(self):
        # 返回数据集的长度
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    def __getitem__(self, index):
        # 获取指定索引的数据项
        if self.data_type == "train":
            return self.random_train_sample()
        else:
            return self.data[index]
        
    def random_train_sample(self):
        # 随机获取训练样本（Anchor, Positive, Negative）
        standard_question_index = list(self.knwb.keys())
        
        # 获取 Anchor 和 Positive 样本
        p = random.choice(standard_question_index)
        if len(self.knwb[p]) < 2:
            return self.random_train_sample()
        else:
            anchor, positive = random.sample(self.knwb[p], 2)
        
        # 获取 Negative 样本
        negative_index = random.choice([index for index in standard_question_index if index != p])
        negative = random.choice(self.knwb[negative_index])
        
        return [anchor, positive, negative]

def load_vocab(vocab_path):
    # 加载词汇表，返回词汇字典
    tocken_dict = {}
    with open(vocab_path, encoding='utf8') as f:
        for index, line in enumerate(f):
            tocken = line.strip()
            tocken_dict[tocken] = index + 1
    return tocken_dict

def load_schema(schema_path):
    # 加载标签schema，返回字典
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())

def load_data(data_path, config, shuffle=True):
    # 加载数据，返回DataLoader
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
