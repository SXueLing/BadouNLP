from config import Config
import torch
import torch.nn as nn
import jieba
import json
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

class DataGenerator:
    def __init__(self, data_path, config, use_bert = True, if_padding = True):
        self.config = config
        self.path = data_path
        self.vocab = self.load_vocab(config['vocab_path'])
        self.bert_vocab = self.load_vocab(config['bert_vocab_path'])
        self.schema = self.load_schema(config['schema_path'])
        self.bert_data = []
        self.data = []
        self.use_bert = use_bert
        self.if_padding = if_padding
        self.load()

    def load_vocab(self, vocab_path):
        # 读取词表txt,返回词-序号，0号位置留给padding
        token_dict = {}
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1
        return token_dict
    
    def load_schema(self, schema_path):
        with open(schema_path, encoding='utf-8') as f:
            return json.load(f)
    
    def load(self):
        '''
        NER的训练数据一行是一个字，已经天然的分成了单字
        利用bert模型文件中携带的词表对应分配id
        from transformers import BertTokenizer适用于处理完整的连续的句子，
        其结果中包含天安门 → ["天", "##安", "##门"]这种子词，不适用NER的训练样本

        我 B-PERSON
        爱 I-PERSON
        北 O
        京 B-LOCATION
        天 I-LOCATION
        安 O
        门 O
        ->
        input_ids =  [101, 2769, 4263, 1266, 776, 1921, 784, 7305, 102]
        labels =     [  8,    2,    6,    8,   0,    4,    8,    8,   8]
        attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        '''
        with open(self.path, 'r', encoding='utf-8') as f:
            segments = f.read().split('\n\n')
            for segment in segments:
                sentence = []
                labels = []
                for line in segment.split('\n'):
                    if line .strip() == '':
                        continue
                    word, label = line.split() # 我 B-PERSON
                    sentence.append(word)      # '我'
                    labels.append(self.schema[label]) # "B-PERSON": 2
                
                if self.use_bert: # 加上CLS和SEP
                    input_ids, attention_mask, label_ids = self.encode_sentence_bert(sentence, labels) # '我'在bert词表中的序号，0/1，ner的label
                    if self.if_padding:
                        max_len = self.config['max_length']
                        input_ids = self.padding(input_ids, max_len, 0)
                        label_ids = self.padding(label_ids, max_len, -1)
                        attention_mask = self.padding(attention_mask, max_len, 0)
                    self.bert_data.append([
                        torch.LongTensor(input_ids),
                        torch.LongTensor(attention_mask),
                        torch.LongTensor(label_ids)
                    ])

                else:
                    input_ids = self.encode_sentence(sentence)
                    if self.if_padding:
                        max_len = self.config['max_length']
                        input_ids = self.padding(input_ids, max_len, 0)
                        label_ids = self.padding(label_ids, max_len, -1)
                    self.data.append([torch.LongTensor(input_ids), torch.LongTensor(label_ids)])
        return
    
    def encode_sentence_bert(self, sentence, labels):
        # 在句子前加上CLS，句子后加上SEP，它俩在NER里视为O
        input_ids = [101] # [CLS]
        label_ids = [8]   # 'O':8
        for char,label in zip(sentence, labels):
            input_ids.append(self.bert_vocab.get(char, self.bert_vocab['[UNK]']))
            label_ids.append(label)
        input_ids.append(102) # [SEP]
        label_ids.append(8) # 'O':8
        attention_mask = [1] * len(input_ids)       
        return input_ids, attention_mask, label_ids
    
    def encode_sentence(self, sentence, padding=True):
        input_id = []
        for char in sentence:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        return input_id
    
    def padding(self, input, max_len, pad_token):
        input = input[:max_len] 
        input += [pad_token] * (max_len - len(input))
        return input
    
    def __len__(self):
        if self.use_bert:
            return len(self.bert_data)
        else:
            return len(self.data)

    def __getitem__(self, index):
        if self.use_bert:
            return self.bert_data[index]
        else:
            return self.data[index]
    
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator(Config["train_data_path"], Config)
    for i in range(5):
        print(dg[i])
        print(dg[i][0].shape)
        print(dg[i][1].shape)
        print(dg[i][2].shape)
