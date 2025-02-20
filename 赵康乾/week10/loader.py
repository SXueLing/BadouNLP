# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import os
import json
from config import Config
import random
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader


'''
生成式任务的输入和输出是错开一位的文本，例'今天天气真好' -> '天天气真好呀'

'''

class DataGenerator:
    '''
    把语料读取成连续的str，按照固定的窗口大小随机取出sentence，
    生成向后错位一位的sentence_target,返回他们的bert-token-id，即模型的输入和输出
    '''
    def __init__(self, config):
        self.config = config
        self.bert_path = config['bert_path']
        self.corpus = ''
        self.corpus_path = config['corpus_path']
        self.data_input = None
        self.data_output = None
        self.data_mask = None
        self.input_max_length = config['input_max_length']
        self.load_corpus()
        # self.build_dataset(config)

    def load_corpus(self):
        with open(self.corpus_path, encoding='gbk') as f:
            for line in f:
                line = line.strip()
                self.corpus += line
        return
    
    def build_sample(self, window_size, corpus):
        start = random.randint(0, len(corpus) - 1 - window_size)
        sentence_input = corpus[start : start + window_size]
        sentence_target = corpus[start + 1: start + window_size + 1] 
        tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        id_input = tokenizer.encode(sentence_input,
                                    padding = 'max_length',
                                    max_length = self.input_max_length,
                                    truncation = True,
                                    return_tensors="pt",
                                    add_special_tokens = False).squeeze(0) #[1, 100] -> [100]
        id_target = tokenizer.encode(sentence_target,
                                    padding = 'max_length',
                                    max_length = self.input_max_length,
                                    truncation = True,
                                    return_tensors="pt",
                                    add_special_tokens = False).squeeze(0)
        mask = torch.tril(torch.ones(window_size, window_size))
        attention_mask = torch.zeros(self.input_max_length, self.input_max_length)
        attention_mask[:window_size, :window_size] = mask
        return (id_input, id_target, attention_mask)
    
    def build_dataset(self, config):
        list_input = []
        list_output = []
        list_mask = []
        batch_size = config['batch_size']
        window_size = config['window_size']
        for i in range(batch_size):
            x,y,m = self.build_sample(window_size, self.corpus)
            list_input.append(x)
            list_output.append(y)
            list_mask.append(m)
        self.data_input = torch.stack(list_input)
        self.data_output = torch.stack(list_output)
        self.data_mask = torch.stack(list_mask)
        return
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return (self.data_input[index], self.data_output[index], self.data_mask[index])

'''
def load_data(config, shuffle=True):
    dg = DataGenerator(config)
    dg.build_dataset(config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
'''

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator(Config)
    dg.build_dataset(Config)
    for i in range(5):
        print(dg[i])
        print(dg[i][0].shape)
        print(dg[i][1].shape)
        print(dg[i][2].shape)
    dg.build_dataset(Config)
    for i in range(5):
        print(dg[i])
        print(dg[i][0].shape)
        print(dg[i][1].shape)
        print(dg[i][2].shape)
