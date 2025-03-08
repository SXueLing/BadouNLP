# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, pre_model_path):
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(pre_model_path)
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                question = line["title"]
                ask = line["content"]
                # tokens1 = self.tokenizer.tokenize(question)
                # tokens2 = self.tokenizer.tokenize(ask)
                # input_ids = self.tokenizer.convert_tokens_to_ids(tokens1 + list(self.tokenizer.sep_token) + tokens2)
                # output_ids = self.tokenizer.convert_tokens_to_ids(tokens2 + list(self.tokenizer.sep_token))
                # output_ids = len(tokens1) * [-100] + output_ids
                tokens1 = self.tokenizer.encode(question, add_special_tokens=False, max_length=20, padding='max_length',
                                                truncation=True)
                tokens2 = self.tokenizer.encode(ask, add_special_tokens=False, max_length=40, padding='max_length',
                                                truncation=True)
                input_id = tokens1 + [103] + tokens2
                output_id = 20 * [0] + tokens2 + [103]
                input_ids = torch.LongTensor(input_id)
                output_ids = torch.LongTensor(output_id)
                self.data.append([input_ids, output_ids, len(question)])
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# 用torch自带的DataLoader类封装数据
def load_data(data_path, batch_size, pre_model_path, shuffle=True):
    dg = DataGenerator(data_path, pre_model_path)
    dl = DataLoader(dg, batch_size=batch_size, shuffle=shuffle)
    return dl


if __name__ == "__main__":
    dg = DataGenerator("valid_tag_news.json", pre_model_path="")
    print(dg[1])
