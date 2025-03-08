# -*- coding: utf-8 -*-

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class DataGenerator:
    def __init__(self, data_path, config, logger):
        self.config = config
        self.logger = logger
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(f'E:\\AIGC\\NLP算法\\bert-base-chinese')  # 加载BERT的分词器
        self.config["vocab_size"] = len(self.tokenizer)         # vocab_size即为BERT分词器的词汇大小
        self.config["pad_idx"] = self.tokenizer.pad_token_id    # PAD的token ID
        self.config["start_idx"] = self.tokenizer.cls_token_id  # [CLS]的token ID
        self.config["end_idx"] = self.tokenizer.sep_token_id    # [SEP]的token ID
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                title = line["title"]
                content = line["content"]
                self.prepare_data(title, content)
        return

    # 使用BERT分词器将文本转换为token_id
    def encode_sentence(self, text, max_length, with_cls_token=True, with_sep_token=True):
        encoding = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=with_cls_token,  # 添加[CLS]和[SEP]标记
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=False
        )
        return encoding['input_ids'].squeeze(0)  # 获取输入ID并去掉batch维度

    # 生成Mask，用于训练中的LM目标
    def generate_lm_mask(self, target_seq, max_length):
        # mask 目标序列中非padding部分
        target_mask = (target_seq != self.config["pad_idx"]).long()
        target_mask = target_mask * (target_seq != self.config["start_idx"]).long()
        target_mask = target_mask * (target_seq != self.config["end_idx"]).long()
        return target_mask

    # 输入输出转化成序列
    def prepare_data(self, title, content):
        # 输入序列
        input_seq = self.encode_sentence(content, self.config["input_max_length"], False, False)
        # 输出序列 (真实标题)
        output_seq = self.encode_sentence(title, self.config["output_max_length"], True, False)
        # 计算mask用于训练
        lm_mask = self.generate_lm_mask(output_seq, self.config["output_max_length"])

        self.data.append([torch.LongTensor(input_seq),
                          torch.LongTensor(output_seq),
                          torch.LongTensor(lm_mask)])

        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, logger, shuffle=True):
    dg = DataGenerator(data_path, config, logger)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config
    dl = load_data(Config["train_data_path"], Config, 1)
    
    for batch_idx, (input_seq, output_seq, lm_mask) in enumerate(dl):
        print(f"Batch {batch_idx + 1}:")
        print(f"input_seq size: {input_seq.size()}")
        print(f"output_seq size: {output_seq.size()}")
        print(f"lm_mask size: {lm_mask.size()}")

        break
