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
        self.config["class_num"] = 2
        if "bert" in self.config["model_type"]:
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.scheme = load_scheme(config["schema_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.task_type = task_type
        self.do_load_data_pre()
        self.change_max_length()
        self.prepare_evaluate_data()
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
        if self.task_type != "train":
            return
        print('加载数据并找到合适的max_len')
        self.do_load_data_pre()
        all_len = [len(q_item) for item in self.json_data for q_item in item['questions']]
        max_len = max(all_len)
        for item in range(1, 100):
            match_lens = [len_item for len_item in all_len if len_item <= math.ceil(item / 100 * max_len)]
            if len(match_lens) / len(all_len) > self.config['len_check']:
                self.config['max_length'] = math.ceil(max_len * item / 100)
                logger.info('当前max_len为%s' % self.config['max_length'])
                change_config_param(self.config['config_path'], max_length=math.ceil(max_len * item / 100))
                break

    def do_load_data_pre(self):
        self.json_data = []
        self.thread_num = 0
        with open(self.path, encoding="utf8") as f:
            for line in f:
                item_data = json.loads(line)
                self.thread_num += 1
                self.json_data.append(item_data)
            f.close()

    def prepare_evaluate_data(self):
        if self.task_type == "train":
            # 这个数据给后面验证使用
            self.cal_data = [(self.encode_sentence_pre(q_item), self.scheme[item['target']]) for item in self.json_data
                             for
                             q_item in item['questions']]

    def random_pairs(self, i_data):
        # 如果列表长度是奇数，去掉最后一个元素
        if len(i_data) % 2 != 0:
            i_data = i_data[:-1]
        # 打乱列表
        random.shuffle(i_data)
        # 每两个元素组成一组
        pairs = [(i_data[i], i_data[i + 1]) for i in range(0, len(i_data), 2)]
        return pairs

    def load(self):
        start_time = time.time()
        self.data = multiprocessing.Manager().list()
        self.test_data = []
        if self.task_type == "train":
            # with multiprocessing.Pool(processes=self.thread_num) as pool:
            #     pool.map(self.build_sample_data, enumerate(self.json_data))
            for index_item, value_item in enumerate(self.json_data):
                self.build_sample_data((index_item, value_item))
            print(f'拿到{len(self.data)}个数据')
            print(f'花费时间{round((time.time() - start_time), 2)}秒')
            random.shuffle(self.data)
        if self.task_type == "test":
            for index, item in enumerate(self.json_data):
                if isinstance(item, list):
                    q = item[0]
                    tag = item[1]
                    input_q_v = self.encode_sentence_pre(q)
                    if tag not in self.scheme:
                        continue
                    tag_value = self.scheme[tag]
                    self.test_data.append([torch.tensor(input_q_v), tag_value])
            random.shuffle(self.test_data)

    def build_sample_data(self, i_data):
        index, item_line = i_data
        retail_datas = [i_value for i_index, i_value in enumerate(self.json_data) if i_index != index]
        random.shuffle(retail_datas)
        # retail_datas = random.sample(retail_datas, 5)
        # 随机取两个元素
        if len(item_line['questions']) < 2:
            return
        n_q_list = [item_r_q for item_r in retail_datas for item_r_q in item_r['questions']]
        for n_q in n_q_list:
            item_c = random.sample(item_line['questions'], 2)
            input_a = item_c[0]
            input_p = item_c[1]
            input_a_v = self.encode_sentence_pre(input_a)
            input_p_v = self.encode_sentence_pre(input_p)
            input_n_q_v = self.encode_sentence_pre(n_q)
            self.data.append([torch.tensor(input_a_v), torch.tensor(input_p_v), torch.tensor(input_n_q_v),
                              (input_a + '-' + input_p + '-' + n_q)])

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
        if self.task_type == "train":
            return len(self.data)
        return len(self.test_data)

    def __getitem__(self, index):
        if self.task_type == "train":
            return self.data[index]
        return self.test_data[index]


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

    Config['task_type'] = 'test'
    dg = DataGenerator(Config['valid_data_path'], Config)
    dl = DataLoader(dg, batch_size=Config["batch_size"], shuffle=True)
    for index, batchdata in enumerate(dl):
        print(batchdata)
    print(dl)
    # Config['max_length'] = 31
