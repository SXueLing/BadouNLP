# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据集 
"""

class Dataset:
    def __init__(self,samples,config):
        self.config = config
        self.samples = samples
        self.vocab = load_vocab(config.get("vocab_path"))
        self.config["vocab_size"] = len(self.vocab)
        self.use_bert = self.config["model_type"] =="bert" or self.config["model_type"] =="bert_lstm"
        if self.use_bert:
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.load()
         

    def load(self):
        self.data = []
        max_length = self.config.get("max_length")
        for sample in self.samples:
            label = sample[0]
            content = sample[1]
            if self.use_bert:
                input_id = self.tokenizer.encode(content, max_length=self.config["max_length"], pad_to_max_length=True)
            else:
                input_id = np.zeros(max_length)
                for i in range(min(len(content), max_length)):
                    input_id[i] = self.vocab.get(content[i], self.vocab.get("[UNK]"))
            input_id = torch.LongTensor(input_id)
            label = torch.LongTensor([label])
            self.data.append([input_id,label])
          
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
    dg = Dataset(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl