# -*- coding: utf-8 -*-

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class NerDataset(Dataset):
    def __init__(self, config, data_path):
        self.data_path = data_path
        self.max_length = config['max_length']
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_path'])
        self.schema = self.load_schema(config['schema_path'])
        self.datas, self.sentences = [], []
        self.load_data()
        return
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        return self.datas[index]
    
    def load_schema(self, schema_path):
        with open(schema_path, encoding='utf8') as f:
            return json.load(f)
        
    def load_data(self):
        with open(self.data_path, encoding='utf8') as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence, labels = [], []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentence))
                inputs = self.tokenizer.encode(
                    sentence, max_length=self.max_length, pad_to_max_length=True
                )
                labels = self.padding(labels, -1)
                self.datas.append(
                    [torch.LongTensor(inputs), torch.LongTensor(labels)]
                )
        return
    
    def padding(self, input, pad_token=0):
        input = [pad_token] + input[: self.max_length-2] + [-1]
        input += [pad_token] * (self.max_length - len(input))
        return input


def data_loader(config, data_path, shuffle=True):
    dataset = NerDataset(config, data_path)
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=shuffle)
