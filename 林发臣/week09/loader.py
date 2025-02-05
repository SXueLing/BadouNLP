# -*- coding: utf-8 -*-

import json
import random
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import logging
import itertools
import math
from nlp_util import change_config_param
import threading
import multiprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config, task_type='train', load_flag=True):
        self.config = config
        self.path = data_path
        self.user_bert = False
        if "bert" in self.config["model_type"]:
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
            self.user_bert = True
        else:
            self.vocab = load_vocab(config["vocab_path"])
            self.config["vocab_size"] = len(self.vocab)
        self.scheme = load_scheme(config["schema_path"])
        self.config["class_num"] = len(self.scheme.keys())
        self.prepare_data = []
        self.data = []
        self.task_type = task_type
        self.sentences = []
        self.do_load_data_pre()
        self.change_max_length()
        if load_flag:
            self.load()

    def encode_sentence_pre(self, input_str):
        if "bert" in self.config["model_type"]:
            input_id = self.tokenizer.encode(input_str, max_length=self.config["max_length"],
                                             pad_to_max_length=True)
        else:
            input_id = self.encode_sentence(input_str)
        return input_id

    def change_max_length(self):
        if self.task_type == 'train':
            print('加载数据并找到合适的max_len')
            all_len = [len(''.join(item[0])) for item in self.prepare_data]
            max_len = max(all_len)
            for item in range(1, 100):
                match_lens = [len_item for len_item in all_len if len_item <= math.ceil(item / 100 * max_len)]
                if len(match_lens) / len(all_len) > self.config['len_check']:
                    self.config['max_length'] = math.ceil(max_len * item / 100)
                    logger.info('当前max_len为%s' % self.config['max_length'])
                    change_config_param(self.config['config_path'], max_length=math.ceil(max_len * item / 100))
                    break

    def do_load_data_pre(self):
        with open(self.path, encoding="utf8") as f:
            all_line = f.read().split('\n\n')
            for line in all_line:
                sentence = []
                label = []
                if self.user_bert:
                    label.append(8)
                for item in line.split('\n'):
                    if item == '':
                        continue
                    str_arr = item.split()
                    sentence.append(str_arr[0])
                    label.append(self.scheme[str_arr[1]])
                label.append(8)
                self.sentences.append(''.join(sentence))
                self.prepare_data.append((sentence, label))
            f.close()

    def load(self):
        for sentence, label in self.prepare_data:
            sentence = [str(i) for i in sentence]
            sentence = ''.join(sentence)
            label = self.padding(label, -1)
            sentence_v = self.encode_sentence_pre(sentence)
            self.data.append([torch.LongTensor(sentence_v), torch.LongTensor(label)])

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
        f.close()
    return token_dict


def load_scheme(scheme_path) -> dict:
    with open(scheme_path, encoding="utf8") as f:
        all_data = f.read()
        scheme_dict = json.loads(all_data)
        f.close()
    return scheme_dict


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, task_type='train', shuffle=True):
    dg = DataGenerator(data_path, config, task_type)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


def get_loader(data_path, config, task_type='train', shuffle=True, load_flag=True):
    dg = DataGenerator(data_path, config, task_type, load_flag=load_flag)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dg, dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator(Config['train_data_path'], Config)
    dl = DataLoader(dg, batch_size=Config["batch_size"], shuffle=True)
    for index, batchdata in enumerate(dl):
        print(batchdata)
    print(dl)
    # Config['max_length'] = 31
