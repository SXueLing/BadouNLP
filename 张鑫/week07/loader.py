# -*- coding: utf-8 -*-

import json

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

import config


class DataGenerator:
    """
    数据加载
    """

    def __init__(self, data_path, config):
        self.data = None
        self.config = config
        self.path = data_path
        self.index_to_label = config["index_to_label"]
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                if line.startswith("0,"):
                    label = 0
                elif line.startswith("1,"):
                    label = 1
                else:
                    continue
                title = line[2:]
                if self.config["model_type"] == "bert":
                    title_index = self.tokenizer.encode(title, max_length=self.config["max_length"],
                                                        pad_to_max_length=True)
                else:
                    title_index = self.encode_sentence(title)
                title_index = torch.LongTensor(title_index)
                label_index = torch.LongTensor([label])
                self.data.append([title_index, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id


def load_vocab(vocab_path):
    """
    加载词表，格式{词: index}
    """
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            # 0留给padding位置，所以从1开始
            token_dict[token] = index + 1
    return token_dict


def load_data(data_path, config, shuffle=True):
    """
    用torch自带的DataLoader类封装数据
    """
    dg = DataGenerator(data_path, config)
    return DataLoader(dg.data, batch_size=config["batch_size"], shuffle=shuffle)


if __name__ == '__main__':
    from config import Config

    print(DataGenerator(Config['valid_data_path'], Config).data)
