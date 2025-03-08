# -*- coding: utf-8 -*-

import json
import random
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import math
from nlp_util import change_config_param
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config, load_flag=True):
        self.config = config
        self.path = data_path
        self.index_to_label = [0, 1]
        self.config["class_num"] = 2
        if "bert" in self.config["model_type"]:
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        if load_flag:
            self.load()

    def check_load_datas(self):
        # 根据数据来设置词句的max_len
        with open(self.config['train_data_all_path'], encoding="utf8") as f:
            lines = f.readlines()
            all_len = [len(line) for line in lines]
            max_len = max(all_len)
            for item in range(1, 100):
                match_lens = [len_item for len_item in all_len if len_item <= math.ceil(item / 100 * max_len)]
                if len(match_lens) / len(all_len) > self.config['len_check']:
                    self.config['max_length'] = math.ceil(max_len * item / 100)
                    logger.info('当前max_len为%s' % self.config['max_length'])
                    change_config_param(self.config['config_path'], max_length=math.ceil(max_len * item / 100))
                    break
            f.close()
        # 处理文件
        # 按比例 compare_pen  来把数据分成两部分 一部分训练使用 一部分测试使用
        with open(self.path, 'r', encoding="utf8") as f:
            lines = f.readlines()
            compare_pen = self.config['compare_pen']
            lines.pop(0)
            bad_lines = [line for line in lines if int(line[0]) == 0]
            good_lines = [line for line in lines if int(line[0]) == 1]
            num1 = math.ceil(len(bad_lines) * compare_pen)
            num2 = math.ceil(len(good_lines) * compare_pen)
            list_bad = random.sample(bad_lines, num1)
            retain_list_bad = [i for i in bad_lines if i not in list_bad]
            list_good = random.sample(good_lines, num2)
            retain_list_good = [i for i in good_lines if i not in list_good]
            all_list = list_bad + list_good
            random.shuffle(all_list)
            with open(self.config['train_data_path'], 'w', encoding="utf8") as train:
                train.writelines(all_list)
                train.close()
            retain_list = retain_list_good + retain_list_bad
            random.shuffle(retain_list)
            with open(self.config['valid_data_path'], 'w', encoding="utf8") as valid:
                valid.writelines(retain_list)
                valid.close()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                words = line.split(',')
                tag = int(words[0])
                title = words[1].replace('\n', '')
                if "bert" in self.config["model_type"]:
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"],
                                                     pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(title)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([tag])
                self.data.append([input_id, label_index])
        return

    def install_data(self):
        self.check_load_datas()

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
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
    return token_dict


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


def install_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config, False)
    dg.install_data()
    logger.info('初始化数据')


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator(Config['train_data_all_path'], Config)
    # print(dg[1])
    # Config['max_length'] = 31
