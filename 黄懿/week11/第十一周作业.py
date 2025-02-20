# coding:utf8
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

"""
基于Bert，实现sft
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
    def forward(self, x, mask=None, y=None):
        if y is not None:
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)


# 加载语料
def load_corpus(path):
    corpus = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            corpus.append([line["title"], line["content"]])
    return corpus


def build_data(tokenizer, corpus, batch_size, max_length):
    data = []
    for i, (prompt, answer) in enumerate(corpus):
        prompt_encode = tokenizer.encode(prompt)
        answer_encode = tokenizer.encode(answer)
        x = prompt_encode + answer_encode
        y = len(prompt_encode) * [-1] + answer_encode
        # 构建mask
        mask = customized_mask(len(prompt_encode), len(answer_encode))
        x = x[:max_length] + [0] * (max_length - len(x))
        y = y[:max_length] + [0] * (max_length - len(y))
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        data.append([x, mask, y])

    return DataLoader(data, batch_size=batch_size, shuffle=True)


def customized_mask(s1_length, s2_length):
    mask = torch.ones(s1_length + s2_length, s1_length + s2_length)
    for i in range(s1_length):
        mask[i, s1_length:] = 0
    for i in range(s2_length):
        mask[s1_length + i, s1_length + i + 1:] = 0
    return mask


# 建立模型
def build_model(char_dim, vocab_size, pretrain_model_path):
    model = LanguageModel(char_dim, vocab_size, pretrain_model_path)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        # 生成文本超过50字则终止迭代
        while len(openings) <= 50:
            x = torch.LongTensor([openings])
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
        prob_distribution = prob_distribution.numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


def main(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    char_dim = 768  # 每个字的维度
    max_length = 50  # 样本文本长度
    vocab_size = 21128  # 字表大小
    learning_rate = 0.001  # 学习率

    pretrain_model_path = r'F:\pretrain_models\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)  # 加载语料
    train_data = build_data(tokenizer, corpus, batch_size, max_length)
    model = build_model(char_dim, vocab_size, pretrain_model_path)  # 建立模型
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, mask, y in train_data:
            optim.zero_grad()  # 梯度归零
            loss = model(x, mask, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("观众热捧沪语版《新闻坊》：每周一期怎么够", model, tokenizer))
        print(generate_sentence("中宣部推动学雷锋 进教材开微博", model, tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    main("sample_data.json", False)
