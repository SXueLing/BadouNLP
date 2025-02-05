# -*- coding: utf-8 -*-

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_path'])
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentenece = []
                labels = [8]
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentenece.append(char)
                    labels.append(self.schema[label])
                sentence = "".join(sentenece)
                self.sentences.append(sentence)
                # a = '俞福柏任院长13年,筹集到海外侨胞捐资1800万元,全部用于发展平民医院的“硬件”建设。'
                # b = list(a)
                # c = self.encode_sentence(a)
                # d = self.encode_sentence(b)
                # input_id = self.encode_sentence(sentence)
                input_ids = self.encode_sentence(sentenece) # 经过实验证明，字符串每个字都独立，会比bert自己的分词效果更好
                # if input_id!=input_ids:
                #     print(input_id,input_ids)
                labels = self.padding(labels, -1)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    def encode_sentence(self, text, padding=True):
        return self.tokenizer.encode(text,
                                     padding="max_length",
                                     max_length=self.config["max_length"],
                                     truncation=True)

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

def load_vocab(vocab_path):
    return BertTokenizer.from_pretrained(vocab_path)

# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)

