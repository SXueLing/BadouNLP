# -*- coding:utf-8 -*-
"""
数据加载
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from test_ai.homework.week07.config.running_config import Config
from test_ai.homework.week07.data.get_sequence_lengths import GetSequenceLengths


def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


class DataGenerator:
    def __init__(self, data_path, config):
        self.token_dict = {}
        self.data = []
        self.data_path = data_path
        self.config = config
        self.index_to_label = {0: 0, 1: 1}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = self.load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.config["max_length"] = GetSequenceLengths(self.data_path).get_length()
        self.load()

    def load_vocab(self, vocab_path):
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                self.token_dict[token] = index + 1
        return self.token_dict

    def load(self):
        df = pd.read_csv(self.data_path)
        for index, col in df.iterrows():
            label = col["label"]
            label = self.label_to_index[label]
            review = col["review"]

            if self.config["model_type"] == "bert":
                input_id = self.tokenizer.encode(review, max_length=self.config["max_length"], padding='max_length',
                                                 truncation=True)
            else:
                input_id = self.encode_sentence(review)
            input_id = torch.LongTensor(input_id)
            label_index = torch.LongTensor([label])
            self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        padding_length = self.config["max_length"] - len(input_id)
        input_id += [0] * padding_length
        return input_id



    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


if __name__ == "__main__":
    dg = DataGenerator("../testdata/valid_data.csv", Config)
    print(dg[1])
    # res = load_data("processed/valid_data.csv", Config)
    # print(res.dataset[1])