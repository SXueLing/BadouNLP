# -*- coding: utf-8 -*-

"""
数据加载
"""
import json
import random

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer


def create_mask(s1, s2, target_shape):
    s1_len = len(s1) + 2
    s2_len = len(s2) + 1
    mask = torch.ones(s1_len + s2_len, s1_len + s2_len)
    for i in range(s1_len):
        mask[i, :s1_len] = 0
    for i in range(s2_len):
        mask[s1_len + i, s1_len + i + 1:] = 0
    # padding
    height, width = mask.shape
    target_height, target_width = target_shape
    result = torch.zeros(target_shape, dtype=mask.dtype, device=mask.device)
    h_end = min(target_height, height)
    w_end = min(target_width, width)
    result[:h_end, :w_end] = mask[:h_end, :w_end]
    return result


class DataGenerator:
    def __init__(self, tokenizer, data_path, config):
        self.corpus = None
        self.schema = None
        self.data = None
        self.config = config
        self.path = data_path
        self.max_length = config['max_length']
        self.tokenizer = tokenizer
        self.load_corpus()
        self.load_data()

    # 加载语料
    def load_corpus(self):
        self.corpus = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.corpus.append([obj['title'], obj['content']])
        return

    # 数据转换
    def load_data(self):
        self.data = []
        for i, (title, content) in enumerate(self.corpus):
            prompt_encode = self.tokenizer.encode(title, add_special_tokens=False)
            answer_encode = self.tokenizer.encode(content, add_special_tokens=False)
            x = self.padding(self.max_length,
                             ([self.tokenizer.cls_token_id] + prompt_encode
                                               + [self.tokenizer.sep_token_id] + answer_encode
                                               + [self.tokenizer.sep_token_id])
                             )
            y = self.padding(self.max_length, (len(prompt_encode) * [-1] + [-1] + answer_encode + [-1]))
            self.data.append([torch.LongTensor(x), torch.LongTensor(y),
                              create_mask(prompt_encode, answer_encode, (self.max_length, self.max_length))])
        return

    def padding(self, max_length, sentence):
        sentence = sentence[:max_length] + [0] * (max_length - len(sentence))
        return sentence

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# 用torch自带的DataLoader类封装数据
def load_data(data_path, tokenizer, config, shuffle=True):
    dg = DataGenerator(tokenizer, data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
