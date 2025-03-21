# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from collections import defaultdict

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: 'B-LOCATION', 1: 'B-ORGANIZATION', 2: 'B-PERSON', 3: 'B-TIME', 
                               4: 'I-LOCATION',5: 'I-ORGANIZATION', 6: 'I-PERSON', 7: 'I-TIME', 8: 'O'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.sentences = []
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                labels = [8] #cls_token
                sentenece = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    #print(line.split())
                    #print('------')
                    char, label = line.split()
                    #print(char, '---', label)
                    sentenece.append(char)
                    labels.append(self.label_to_index[label])
                sentence = "".join(sentenece)
                self.sentences.append(sentence)
                
                
                #print(self.sentences, '------', labels)
                # if self.config["model_type"] == "bert":
                #     input_id = self.tokenizer.encode(sentence, max_length=self.config["max_length"], pad_to_max_length=True)
                # else:
                #     input_id = self.encode_sentence(sentence)
                input_ids = self.encode_sentence(sentence)
                labels = self.padding(labels, -1)
                #print(input_id, '---', labels)
                input_ids = torch.LongTensor(input_ids)
                label_index = torch.LongTensor([labels])
                self.data.append([input_ids, label_index])
        return 

    def encode_sentence(self, text, padding=True):
        # input_id = []
        # for char in text:
        #     input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        # input_id = self.padding(input_id)
        # return input_id
        return self.tokenizer.encode(text, padding="max_length", max_length=self.config["max_length"], truncation=True)

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def decode(self, sentence, labels):
        sentence = "$" + sentence
        labels = "".join([str(x) for x in labels[:len(sentence)+2]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            print("location", s, e)
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            print("org", s, e)
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            print("person", s, e)
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            print("time", s, e)
            results["TIME"].append(sentence[s:e])
        return results
    
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
    from config_ner import Config
    dg = DataGenerator("/Users/jessicachan/nlp20/ner/ner_data/train", Config)
    print(dg[1])
