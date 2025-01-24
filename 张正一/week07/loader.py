import json
import re
import os
import torch
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from config import Config

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.data_path = data_path
        self.index_to_label = {0: '差评', 1: '好评'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config['class_num'] = len(self.index_to_label)
        if self.config['model_type'] == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(self.config['pretrain_model_path'])
        self.vocab = load_voacb(self.config['vocab_path'])
        self.config['vocab_size'] = len(self.vocab)
        self.load()
        
    def load(self):
        self.data = []
        self.test_data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)
            test_ratio = 0.2
            train_data = list(csv_reader)
            tag_0_train_data = list(filter(lambda x: x[0] == '0', train_data))
            tag_1_train_data = list(filter(lambda x: x[0] == '1', train_data))
            tag_0_test_count = int(len(tag_0_train_data) * test_ratio)
            tag_1_test_count = int(len(tag_1_train_data) * test_ratio)
            print('数据集标签0的数量：', len(tag_0_train_data))
            print('数据集标签1的数量：', len(tag_1_train_data))
            print('测试集标签0的数量：', tag_0_test_count)
            print('测试集标签1的数量：', tag_1_test_count)
            for index, line in enumerate(tag_0_train_data):
                tag, review = line
                if self.config['model_type'] == 'bert':
                    input_ids = self.tokenizer.encode(review, max_length=self.config['max_length'], pad_to_max_length=True)
                else:
                    input_ids = self.encode_sentence(review)
                input_ids = torch.LongTensor(input_ids)
                tag = torch.LongTensor([int(tag)])
                if index < tag_0_test_count:
                    self.test_data.append([input_ids, tag])
                else:
                    self.data.append([input_ids, tag])

            for index, line in enumerate(tag_1_train_data):
                tag, review = line
                if self.config['model_type'] == 'bert':
                    input_ids = self.tokenizer.encode(review, max_length=self.config['max_length'], pad_to_max_length=True)
                else:
                    input_ids = self.encode_sentence(review)
                input_ids = torch.LongTensor(input_ids)
                tag = torch.LongTensor([int(tag)])
                if index < tag_1_test_count:
                    self.test_data.append([input_ids, tag])
                else:
                    self.data.append([input_ids, tag])
        print(38, len(self.data), len(self.test_data))
    
    def encode_sentence(self, sentence):
        input_ids = []
        for word in sentence:
            input_ids.append(self.vocab.get(word, self.vocab['[UNK]']))
        input_ids = self.padding(input_ids)
        return input_ids
     
    # 截断和填充
    def padding(self, input_ids):
        input_ids = input_ids[:self.config['max_length']]
        input_ids += [0] * (self.config['max_length'] - len(input_ids))
        return input_ids

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def get_test_data(self):
        return self.test_data

class TestDataGenerator(DataGenerator):
    def __init__(self, data_path, config):
        super().__init__(data_path, config)
        
    def __len__(self):
        return len(self.test_data)
    
    def __getitem__(self, index):
        return self.test_data[index]
                
    
def load_voacb(vocab_path):
    token_dict = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1
    return token_dict
    
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    print(dg)
    dl = DataLoader(dg, batch_size=config['batch_size'], shuffle=shuffle)
    return dl  

def load_test_data(data_path, config, shuffle=True):
    dg = TestDataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config['batch_size'], shuffle=shuffle)
    return dl

if __name__ == '__main__':
    dg = DataGenerator('文本分类训练数据.csv', Config)