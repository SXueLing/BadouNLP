# -*- coding: utf-8 -*-

import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


class SentenceMatchDataset(Dataset):
    def __init__(self, config, mode='train'):
        assert mode == 'train' or mode == 'test'
        self.config = config
        self.mode = mode
        self.data_path = config['train_data_path'] if mode == 'train' else config['valid_data_path']
        self.vocab = load_vocab(config['vocab_path'])
        self.config['vocab_size'] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        self.epoch_data_size = config['epoch_data_size']
        self.load()


    def __len__(self):
        if self.mode == 'train':
            return self.epoch_data_size
        else:
            return len(self.data)
        
    def __getitem__(self, index):
        if self.mode == 'train':
            return self.random_train_sample()
        else:
            return self.data[index]
    
    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.data_path, encoding='utf8') as f:
            for line in f:
                line = json.loads(line)
                if self.mode == 'train':
                    assert isinstance(line, dict)
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                else:
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])
        return
    
    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id
    
    def padding(self, input_id):
        input_id = input_id[:self.config["max_sentence_len"]]
        input_id += [0] * (self.config["max_sentence_len"] - len(input_id))
        return input_id
    
    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())
        key1, key2 = random.sample(standard_question_index, 2)
        if len(self.knwb[key1]) < 2:
            return self.random_train_sample()
        
        a, p = random.sample(self.knwb[key1], 2)
        n = random.choice(self.knwb[key2])
        return [a, p, n]
        

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1
    return token_dict


def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())


def load_data(config, mode, shuffle=True):
    dataset = SentenceMatchDataset(config, mode)
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=shuffle)
