# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

"""
数据加载
"""
from transformers import BertTokenizer
def collate_fn(batch):
# 获取 batch 中的最大序列长度
    max_len = max([len(item[0]) for item in batch])  # 计算最大的input_ids长度

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        input_id = item[0]
        attention_mask_item = item[1]
        label = item[2]

        # 计算padding的长度
        padding_len = max_len - len(input_id)

        # 填充 input_ids 和 attention_mask
        input_id_padded = torch.cat([input_id, torch.zeros(padding_len, dtype=torch.long)])  # padding token_id 通常是0
        attention_mask_padded = torch.cat([attention_mask_item, torch.zeros(padding_len, dtype=torch.long)])  # 填充注意力掩码

        # 填充 labels
        label_padded = torch.cat([label, torch.full((padding_len,), -1, dtype=torch.long)])  # 填充标签，通常使用-1作为padding标签

        # 将每个处理后的样本添加到列表中
        input_ids.append(input_id_padded.unsqueeze(0))  # 保持每个样本的维度为 [1, seq_len]
        attention_mask.append(attention_mask_padded.unsqueeze(0))  # 同上
        labels.append(label_padded.unsqueeze(0))  # 同上

    # Stack 所有的张量以形成一个批次
    input_ids = torch.cat(input_ids, dim=0)  # shape: [batch_size, max_len]
    attention_mask = torch.cat(attention_mask, dim=0)  # shape: [batch_size, max_len]
    labels = torch.cat(labels, dim=0)  # shape: [batch_size, max_len]

    return input_ids, attention_mask, labels
    

class DataGenerator(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])  # 使用BERT的tokenizer
        self.config["vocab_size"] = len(self.tokenizer)  # 使用BERT的词汇表大小
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
                labels = self.padding(labels, -1)
                self.data.append([input_ids,torch.LongTensor(labels)])

    def encode_sentence(self, text, padding=True):
        encoding = self.tokenizer("".join(text), padding='max_length', truncation=True, max_length=self.config["max_length"], return_tensors='pt')

        input_id = encoding['input_ids'].squeeze(0)  

        return input_id

    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 返回的data应包含 input_ids, attention_mask 和 labels
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

# 加载数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config
    # dg = DataGenerator("./ner_data/train", Config)
    train_data = load_data("./ner_data/train", Config)
    for index, batch_data in enumerate(train_data):
        input_id, labels = batch_data
        print("input_id:",input_id)
        print("labels:",labels)
        if index == 2:
            break
