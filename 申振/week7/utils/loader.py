# Placeholder for data loading and preprocessing
import csv
import json

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset
from transformers import BertTokenizer


class ParseCsv:
    def __init__(self, csv_path,config):
        self.config = config
        self.csv_path = csv_path
        self.train_data_path = config['train_data_path']
        self.valid_data_path = config['valid_data_path']
        self.load()
    def load(self):
        self.pred = []
        self.data = []
        with open(self.csv_path, mode='r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader)
            print("Headers:", headers)
            for row in csv_reader:
                self.pred.append(row[0])
                self.data.append(row[1])

        data_train, data_val, pred_train, pred_val = train_test_split(self.data, self.pred,
                                                          test_size=self.config['valid_size'],
                                                          random_state=self.config['seed'],
                                                          stratify=self.pred)
        train_data = []
        len_data = len(data_train)
        print("len_data:", len_data)
        for i in range(len_data):

            train_data.append(
                {
                    headers[1]: data_train[i],
                    headers[0]: pred_train[i]
                }
            )


        with open(self.train_data_path, 'w', encoding='utf-8') as train_data_file:
            for data in train_data:
                train_data_file.write( json.dumps(data, ensure_ascii=False) + "\n")

        val_data = []
        len_data = len(data_val)
        for i in range(len_data):
            print(data_val[i])
            val_data.append(
                {
                    headers[1]: data_val[i],
                    headers[0]: pred_val[i]
                }
            )
        with open(self.valid_data_path, 'w', encoding='utf-8') as valid_data_file:
            for data in val_data:
                valid_data_file.write( json.dumps(data, ensure_ascii=False) + "\n")



class DataGenerator():
    def __init__(self,data_path,config):
        self.config = config
        self.path = data_path
        self.index_to_label = config['index_to_label']
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config['model_type'] == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(self.config['pretrain_model_path'])
        self.vocab = load_vocab(self.config['vocab_path'])
        self.config['vocab_size'] = len(self.vocab)
        self.load()
    def load(self):
        self.data = []
        print(self.path)
        with open(self.path,'r') as f:
            for line in f:
                line = json.loads(line)
                label = line['label']
                label = self.label_to_index[label]
                review = line['review']
                if self.config['model_type'] == 'bert':
                    tokens = self.tokenizer.encode(review,max_length=self.config['max_length'],pad_to_max_length=True)
                else:
                    tokens = self.encode_sentence(review)
                tokens = torch.LongTensor(tokens)
                label_index = torch.LongTensor([label])
                self.data.append([tokens, label_index])
    def encode_sentence(self, sentence):
        tokens = []
        for token in sentence:
            tokens.append(self.vocab.get(token,self.vocab["[UNK]"]))
        tokens = self.padding(tokens)
        return tokens
    def padding(self,tokens):
        tokens = tokens[:self.config['max_length']]
        tokens += [0] * ((self.config['max_length'] - len(tokens)) - 1)
        return tokens
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# 加载词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1
    return token_dict

# 加载数据
def load_data(data_path,config, shuffle=True):
    dg = DataGenerator(data_path,config)
    dl = DataLoader(dg, batch_size=config["batch_size"],shuffle=shuffle)
    return dl
