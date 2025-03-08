import json

import torch
from transformers import BertTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.data = []
        self.data_path = data_path
        self.config = config
        self.index_to_label = {0: "差评", 1: "好评"}
        self.config["class_num"] = len(self.index_to_label)
        print(self.config['class_num'])
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(self.config["pretrain_model_path"])
        self.vocab = load_vocab(self.config['vocab_path'])
        self.config["vocab_size"] = len(self.vocab)
        self.load()

    def load(self):
        #         读取 xxx.json
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for index, item in enumerate(data):
                label = item.get("label", None)
                review = item.get("review", None)
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(review, max_length=self.config["max_length"], pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(review)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])

    def encode_sentence(self, sentence):
        input_id = []
        for char in sentence:
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

def gen_data():
    data_path = "data/文本分类练习.csv"
    df = pd.read_csv(data_path)
    train_data = []
    test_data = []
    # 均衡正负样本
    a_data = []
    b_data = []
    for index, row in df.iterrows():
        label = row['label']
        if label == 0:
            if len(a_data) < 4000:
                a_data.append(row)
        elif label == 1:
            if len(b_data) < 4000:
                b_data.append(row)
    # 将数据划分80%到train_data，20%到test_data
    index = 0
    for a, b in zip(a_data, b_data):
        data_a = {}
        data_b = {}
        data_a["label"] = a.get("label")
        data_a["review"] = a.get("review")
        data_b["label"] = b.get("label")
        data_b["review"] = b.get("review")
        if index % 5 == 0:
            test_data.append(data_a)
            test_data.append(data_b)
        else:
            train_data.append(data_a)
            train_data.append(data_b)
        index += 1
    # 保存train_data到data/train_data.json 文件中
    json.dump(train_data, open("data/train_data.json", "w", encoding="utf8"), ensure_ascii=False)
    json.dump(test_data, open("data/test_data.json", "w", encoding="utf8"), ensure_ascii=False)


def text_avg_len():
    data_path = "data/文本分类练习.csv"
    df = pd.read_csv(data_path)
    text_len = 0
    for index, row in df.iterrows():
        text_len += len(row['review'])
    return text_len / len(df)

if __name__ == '__main__':
    # gen_data()
    text_avg_len = text_avg_len()
    print(text_avg_len)
    # from config import Config

    # dg = DataGenerator("data/test_data.json", Config)
    # print(dg[1])
