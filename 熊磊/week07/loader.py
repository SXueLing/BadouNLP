import os
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer
from config import Config


class DataGenerator(Dataset):
    def __init__(self, Config):
        super().__init__()
        self.Config = Config
        self.data_path = Config['data_path']
        self.model_type = Config['model_type']
        self.tokenizer = BertTokenizer.from_pretrained(Config['bert_path'])
        self.max_length = Config['max_length']
        self.vocab = load_vocab(Config)
        self.load()

    def load(self):
        self.data = []
        data_csv = pd.read_csv(self.data_path)

        for i in range(len(data_csv)):
            label = data_csv['label'][i]
            sentence = data_csv['review'][i]
            if self.model_type == "bert":
                sentence = self.tokenizer.encode(sentence, max_length=self.max_length, pad_to_max_length=True)
            else:
                sentence = self.encode(sentence)
                sentence = self.padding(sentence)

            self.data.append([torch.LongTensor(sentence), torch.LongTensor([label])])
    
    def encode(self, sentence):
        return [self.vocab.get(x, self.vocab["[UNK]"]) for x in sentence]
    
    def padding(self, sentence):
        return sentence[:self.max_length] + [0] * (self.max_length - len(sentence))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        return self.data[item]
    

def load_vocab(Config):
    vocab = {}
    with open(Config['vocab_path'], encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            vocab[token] = index + 1
    Config['vocab_size'] = len(vocab)
    return vocab

def load_data(Config, shuffle = True):
    dg = DataGenerator(Config)
    train_size = int(0.8 * len(dg))
    val_size = len(dg) - train_size
    train_dataset, val_dataset = random_split(dg, [train_size, val_size])

    dt = DataLoader(train_dataset, batch_size=Config['batch_size'], shuffle=shuffle)
    dv = DataLoader(val_dataset, batch_size=Config['batch_size'], shuffle = shuffle)
    return dt, dv


def decode(sentence):
    tokenizer = BertTokenizer.from_pretrained(Config['bert_path'])
    print(tokenizer.decode(sentence))

if __name__ == '__main__':
    from config import Config
    dg = DataGenerator(Config)
    print(dg[1])
    decode(dg[1][0])
