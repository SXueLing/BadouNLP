# -*- coding: utf-8 -*-

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""

class DataGenerator(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.schema = self.load_schema(config["schema_path"])
        self.sentences = []
        self.data = []
        self.load()

    def load(self):
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentence))
                input_ids, attention_mask, label_ids = self.encode_sentence(sentence, labels)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.LongTensor(label_ids)])

    def encode_sentence(self, text, labels):
        tokens = []
        label_ids = []
        for char, label in zip(text, labels):
            tokenized_char = self.tokenizer.tokenize(char)
            tokens.extend(tokenized_char)
            # If a character is split into multiple sub-tokens, we assign the same label to all sub-tokens
            label_ids.extend([label] * len(tokenized_char))

        # Truncate or pad the sequences
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = self.padding(input_ids)
        attention_mask = [1] * len(input_ids)
        attention_mask = self.padding(attention_mask, pad_token=0)
        label_ids = self.padding(label_ids, pad_token=-1)

        return input_ids, attention_mask, label_ids

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1], self.data[index][2], self.sentences[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)



