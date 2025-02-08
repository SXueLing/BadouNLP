# -*- coding: utf-8 -*-
import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertTokenizer


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"]) # chars.txt
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding='utf8') as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentence))
                # tokenizer生成id
                if self.config["model_type"] == "bert":
                    # 不足的补0
                    # input_ids = self.tokenizer.encode(sentence, max_length=self.config["max_length"], pad_to_max_length=True)
                    # labels = [8] # 补上token对应的输出为8，token也会参与训练
                    # labels = self.padding(labels, -1)
                    # # 如果长度超过100，最后一位的输入是[sep],输出改成8
                    # if len(labels) == self.config["max_length"] and labels[self.config["max_length"]-1] != -1:
                    #     labels[self.config["max_length"]-1] = 8

                    # token也参与
                    input_ids = self.encode_sentence(sentence)
                    labels = self.padding(labels, -1) # 输出值用-1作为padding
                else:
                    input_ids = self.encode_sentence(sentence)
                    labels = self.padding(labels, -1) # 输出值用-1作为padding
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return
    
    # bert不走这里
    def encode_sentence(self, text, padding=True):
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
    
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding='utf8') as f:
            return json.load(f)

# 加载词表
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
    # print(len(dg[0][0]), 'len(dg)') # 100， len(dg)=1412
    # print(dg[0][0].tolist().index(0), dg[0][1].tolist().index(-1))
    # print(dg[5][0], dg[5][1], 'dg[0]')
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    # dg = DataGenerator("ner_data/train", Config)
    dg2 = load_data("ner_data/test", Config)
