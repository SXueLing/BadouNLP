# -*- coding: utf-8 -*-
# @Time    : 2025/1/8 16:10
# @Author  : yeye
# @File    : loader.py
# @Software: PyCharm
# @Desc    :
import pandas as pd
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.label_to_index = {0: "差评", 1: "好评"}
        self.config["class_num"] = len(self.label_to_index)
        self.load()

    def load(self):
        self.data = []
        df = pd.read_csv(r"D:\code\pycharm\NLP\week7\week7\train_data.csv")
        for label, review in zip(df["label"], df["review"]):
            if self.config["model_type"] == "bert":
                input_id = self.tokenizer.encode(review, max_length=self.config["max_length"], pad_to_max_length=True)
            else:
                input_id = self.encode_sentence(review)
            input_id = torch.LongTensor(input_id)
            label = torch.LongTensor([label])
            self.data.append([input_id, label])
        return

    def encode_sentence(self, review):
        input_id = [self.vocab.get(char, self.vocab['[UNK]']) for char in review]
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    vocab_dict = {}
    with open(vocab_path, encoding='utf-8') as f:
        for index, line in enumerate(f):
            token = line.strip()
            vocab_dict[token] = index + 1
    return vocab_dict


def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator(r"D:\code\pycharm\NLP\week7\data\train_tag_news.json", Config)
    print(dg[1])
