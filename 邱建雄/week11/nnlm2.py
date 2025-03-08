# coding:utf8
import json

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re

from torch.utils.data import TensorDataset
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)

        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')

        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, mask=None):
        if y is not None:
            # 训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # 预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)


# 加载语料
def load_corpus(path):
    corpus = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            title = line["title"]
            content = line["content"]
            corpus.append((title, content))
    return corpus


def build_dataset(sample_length, tokenizer, batch_size, corpus):
    dataset = []
    for i, (title, content) in enumerate(corpus):
        x_a = tokenizer.encode(title, add_special_tokens=False)
        y_a = tokenizer.encode(content, add_special_tokens=False)
        x = [tokenizer.cls_token_id] + x_a + [tokenizer.sep_token_id] + y_a + [tokenizer.sep_token_id]
        y = [-1] * (len(x_a) + 1) + y_a + [tokenizer.sep_token_id] + [-1]
        x = x[:sample_length] + [0] * (sample_length - len(x))
        y = y[:sample_length] + [0] * (sample_length - len(y))
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        mask = torch.tril(torch.ones(len(x), len(x)))
        for i_1 in range(len(x_a) + 2):
            for j in range(len(x_a) + 2):
                mask[i_1, j] = 1
        #print(mask.shape)
        dataset.append([x, y, mask])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


# 建立模型
def build_model(vocab, char_dim, pretrain_model_path):
    model = LanguageModel(768, 21128, pretrain_model_path)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, tokenizer, window_size):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        # 生成了换行符，或生成文本超过30字则终止迭代
        while len(openings) <= 50:
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            openings.append(index)
    return tokenizer.decode(openings)


def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


def train(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    char_dim = 768  # 每个字的维度
    window_size = 100  # 样本文本长度
    vocab_size = 21128  # 字表大小
    learning_rate = 0.001  # 学习率

    pretrain_model_path = r'C:\Users\Administrator\Desktop\人工智能\week6 语言模型和预训练\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)  # 加载数据
    train_data = build_dataset(window_size, tokenizer, batch_size, corpus)

    model = build_model(vocab_size, char_dim, pretrain_model_path)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, y, mask in train_data:
            if torch.cuda.is_available():
                x, y, mask = x.cuda(), y.cuda(), mask.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, y, mask)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("阿根廷歹徒抢服装尺码不对拿回店里换", model, tokenizer, window_size))
        #print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train(r"sample_data.json", False)
