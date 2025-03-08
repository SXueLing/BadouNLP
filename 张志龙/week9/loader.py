# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
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
        # self.vocab = load_vocab(config["vocab_path"])
        self.tokenizer = load_vocab_bert(config["bert_path"])
        # self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")  # 双换行分句
            for segment in segments:
                # bert在做encoding的时候，不特别声明会默认在句子的前面加上cls_token
                # 序列标注任务需要输入和输出是对齐的，所以需要在label序列的前面补一个cls_token对应的结果，可以对应到一个无关字的label上
                # 所以一开始sentences是空的，labels先放了个8进去，8代表一个无关字$
                # 加上了$之后，sentences和labels就对齐了，最终解码的时候就得注意现在labels是多了一位，所以输入sentence的前面得补上$占位
                # 事实上结尾也会自动加上一个sep_token，所以最终的输入是[cls]句子[sep]，输出是[8]标签[sep]，但是结尾在解码的时候可以不需要考虑
                # 所以这里就没有对结尾进行特殊处理
                sentenece = []
                labels = [8]  # cls_token
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentenece.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentenece))
                # 通过jieba分词，将句子转换为id序列
                # 由于是序列标注任务，所以需要将每个字符的标签对应到每个字符上
                # 如果使用bert等预训练模型，可以将句子转换为token序列，然后将标签对应到token上
                # input_ids = self.encode_sentence(sentenece)
                input_ids = self.bert_encode_sentence(sentenece)
                labels = self.padding(labels, -1)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    def bert_encode_sentence(self, text, padding=True):
        return self.tokenizer.encode(text, 
                                     padding="max_length",
                                     max_length=self.config["max_length"],
                                     truncation=True)
    def encode_sentence(self, text, padding=True):
        """
        该函数将输入的文本转换为id序列
        :param text: 输入文本
        :param padding: 是否进行padding
        :return: id序列
        实现方式：将文本中的每个字符转换为id，如果字符不在字表中，则转换为[UNK]的id
        """
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
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

# 加载字表或词表
def load_vocab(vocab_path):
    """
    加载字表或词表
    :param vocab_path: 字表或词表路径
    """
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

def load_vocab_bert(vocab_path):
    """
    加载bert的词表
    :param vocab_path: 词表路径
    """
    return BertTokenizer.from_pretrained(vocab_path)

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)

