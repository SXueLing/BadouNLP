# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
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
                input_ids = self.encode_sentence(sentence)
                labels = self.padding(labels, pad_token=-500)  # 使用-500作为padding
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    def encode_sentence(self, text, padding=True):
        input_id = []
        if self.config["use_bert"]:  # 如果使用BERT
            input_id.append(self.vocab["[CLS]"])
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
            input_id.append(self.vocab["[SEP]"])
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
        input_id, label = self.data[index]
        attention_mask = (input_id > 0).long()  # 假设0是padding
        return input_id, attention_mask, label
        #return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index  # 直接使用index，不要加1
    return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle, collate_fn=collate_fn)
    return dl
    # dg = DataGenerator(data_path, config)
    # dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    # return dl

def collate_fn(batch_data):
    batch_input_ids = []
    batch_attention_masks = []
    batch_labels = []
    for input_id, attention_mask, label in batch_data:
        batch_input_ids.append(input_id)
        batch_attention_masks.append(attention_mask)
        batch_labels.append(label)
    return torch.stack(batch_input_ids), torch.stack(batch_attention_masks), torch.stack(batch_labels)




if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)

