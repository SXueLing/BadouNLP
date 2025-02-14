# -*- coding: utf-8 -*-
# @Date    :2025-02-11 22:12:43
# @Author  :SongWei (songweiof@gmail.com)
# @Link    :https://github.com/Finder12345
# @Version :python 3.11
# @Software:Sublime Text


import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset,random_split
from transformers import BertTokenizer


class MyDataSet(Dataset):
    """docstring for DataGenerator"""

    def __init__(self, config):
        super(MyDataSet, self).__init__()
        self.data_path = config["data_path"]
        self.config = config
        self.index_to_label = {0: "积极评论", 1: "消极评论"}
        self.label_to_index = dict(
            ((label, index) for index, label in self.index_to_label.items()))
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(
                config["pretrain_model_path"])
        self.vocab_to_index_dict = self.load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab_to_index_dict)
        self.load()

    def load(self):
        # 文本的预处理阶段，将文本转化为token IDs格式
        self.data = []
        df = pd.read_csv(self.data_path)
        for index, row in df.iterrows():
            label = row["label"]
            review = row["review"]
            if self.config["model_type"] == "bert":
                # [1,len]
                token_ids = self.tokenizer.encode(review,
                                                  max_length=self.config["max_length"],
                                                  padding="max_length")
            else:
                token_ids = self.encode_sentence(review)
            token_ids = torch.LongTensor(token_ids)
            label_index = torch.LongTensor([label])  # 实现样本与标签的表征维度一致
            self.data.append([token_ids, label_index])
        

    def encode_sentence(self, text):
        token_ids = []
        for char in text:
            token_ids.append(self.vocab_to_index_dict.get(
                char, self.vocab_to_index_dict["[UNK]"]))
        token_ids = self.padding(token_ids)
        return token_ids

    def padding(self, token_ids):
        token_ids = token_ids[:self.config["max_length"]]
        token_ids += [0]*(self.config["max_length"]-len(token_ids))
        return token_ids

    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1  # 0的位置留个padding
        return token_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_IDs, label = self.data[idx]
        return token_IDs, label


def load_data(config, shuffle=True):
    dataset = MyDataSet(config)
    train_rate,val_rate = config["train_rate"],1-config["train_rate"]


    train_dataset,val_dataset = random_split(dataset,[train_rate,val_rate])

    train_dataloader = DataLoader(dataset=train_dataset,
                            batch_size=config["batch_size"],shuffle=shuffle)
    val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=config["batch_size"],shuffle=shuffle)
    return train_dataloader,val_dataloader



if __name__ == '__main__':
    data_path = "../文本分类练习.csv"
    import config
    # mydataset = MyDataSet(config.Config)
    train_data,val_data = load_data(config.Config)

    
