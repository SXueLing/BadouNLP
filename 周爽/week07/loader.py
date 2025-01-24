# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import csv
import random
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '差评', 1: '好评'}
        # self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                label = line['label']
                review = line['review']
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(review, max_length=self.config["max_length"], pad_to_max_length=True)
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
    #训练集/验证集(9:1)
    data = []
    corrects_sum = 0 #正样本数量
    wrong_sum = 0#负样本数量
    review_len = []

    with open(r"D:\000nlpStudy\week7文本分类问题\data\文本分类练习.csv", encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            if(row[0] != 'label'):
                label_obj = {}
                lable = int(row[0])
                if(lable == 1):
                    corrects_sum += 1
                else:
                    wrong_sum += 1
                review_len.append(len(row[1]))
                label_obj['label'] = lable
                label_obj['review'] = row[1]
                json_obj = json.dumps(label_obj, ensure_ascii=False)
                data.append(json_obj)
    #数据分析:正负样本数，文本平均长度等
    print("正样本数：", corrects_sum)
    print("负样本数：", wrong_sum)
    print("文本平均长度：", sum(review_len)/len(review_len))
    print("文本平均长度：", np.mean(review_len))

    random.shuffle(data)

    train_data = data[:int(len(data)*0.9)]
    valid_data = data[int(len(data)*0.9):]

    #写入文件:训练集/验证集
    file_train_data = "train_data.json"
    file_valid_data = "valid_data.json"

    with open(file_train_data, "w", encoding="utf8") as f:
        f.write("\n".join(train_data))
    with open(file_valid_data, "w", encoding="utf8") as f:
        f.write("\n".join(valid_data))

    dg = DataGenerator(r"D:\000nlpStudy\week7文本分类问题\data\valid_data.json", Config)
    print(dg[0])

