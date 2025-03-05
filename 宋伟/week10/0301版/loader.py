# -*- coding: utf-8 -*-
# @Date    :2025-03-01 22:57:43
# @Author  :SongWei (songweiof@gmail.com)
# @Link    :https://github.com/Finder12345
# @Version :python 3.11
# @Software:Sublime Text


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import random


class MyDataSet(object):
    """docstring for MyDataset"""

    def __init__(self, config):
        super(MyDataSet, self).__init__()
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_path'])

        # 将字表进行修改即可
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)

        self.corpus = load_corpus(config["corpus_path"])
        self.load()

    def load(self):
        self.data = []  # 记录总的数据条数，并每条数据以字典形式进行存储
        for _ in range(self.config["sample_length"]):
            data = {}
            x, y = build_sample(
                self.tokenizer, self.config['window_size'], self.corpus)
            data["inputs"] = x.squeeze(0)
            data["labels"] = y.squeeze(0)
            self.data.append(data)
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_vocab(path)->dict:
    vocab = {"<pad>": 0}
    with open(path, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            char = line.strip()
            vocab[char] = idx + 1
    return vocab


def load_corpus(path)->str:
    corpus = ""
    with open(path, encoding='gbk') as f:
        for line in f:
            corpus += line.strip()
    return corpus


def build_sample(tokenizer, window_size, corpus)->tuple:
    start = random.randint(0, len(corpus)-1 - window_size)
    end = start + window_size
    window_chars = corpus[start:end]
    traget_chars = corpus[start+1:end+1]

    # x = [vocab.get(char,vocab["<UNK>"]) for char in window_chars]
    # y = [vocab.get(char,vocab["<UNK>"]) for char in traget_chars]
    x = tokenizer.encode(window_chars,
                     max_length=10,
                     return_tensors='pt',
                     padding='max_length',
                     truncation=True,
                     add_special_tokens=False)


    y = tokenizer.encode(traget_chars,
                     max_length=10,
                     return_tensors='pt',
                     padding='max_length',
                     truncation=True,
                     add_special_tokens=False)
    return x, y


def data_load(config, shuffle=True)->DataLoader:
    dataset = MyDataSet(config)
    dataloder = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=shuffle)
    return dataloder


if __name__ == '__main__':
    from config import Config
    dataset = MyDataSet(Config)
    print(dataset[0])

    quit()
    dataloader = data_load(Config)
    for i, batch_data in enumerate(dataloader):
        print(batch_data)  # [batch,seq_len],[16,10]

        break
