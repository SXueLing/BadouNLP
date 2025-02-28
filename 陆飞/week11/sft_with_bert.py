# coding:utf8

import json
import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re

from transformers import BertModel, BertTokenizer

"""
bert实现sft
"""


class LanguageModel(nn.Module):
    def __init__(self):
        super(LanguageModel, self).__init__()
        self.layer = BertModel.from_pretrained(r"/home/lufei/AI/bert-base-chinese", return_dict=False)
        self.classify = nn.Linear(self.layer.config.hidden_size, 21128)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, q_len, x, y=None):
        mask = []
        for sen, qlen in zip(x, q_len):
            # sen = 'sdadsadsadsa'
            # qlen = 5
            matrix = [[1 for _ in range(len(sen))] for _ in range(len(sen))]
            for i in range(qlen):
                for j in range(qlen, len(sen)):
                    matrix[i][j] = 0
            for i in range(qlen, len(sen)):
                for j in range(i + 1, len(sen)):
                    matrix[i][j] = 0
            mask.append(matrix)
            # print(matrix)
        # print(mask)
        mask = torch.LongTensor(mask)

        if torch.cuda.is_available():
            mask = mask.cuda()
        if y is not None:
            x, _ = self.layer(x, attention_mask=mask)  # output shape:(batch_size, sen_len, hidden_size)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1), ignore_index = 0)
        else:
            x, _ = self.layer(x)  # output shape:(batch_size, sen_len, hidden_size)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)

# 加载语料
def load_QAs(path):
    lines = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            lines.append(line)
    return lines


# 随机生成一个样本
def build_sample(tokenizer, window_size, qas):
    index = random.randint(0, len(qas) - 1)
    qa = qas[index]
    window = (qa["title"] + qa["content"])[:window_size]
    target = window[1 : len(window)]
    x = tokenizer.encode(window, add_special_tokens=False, max_length=window_size, pad_to_max_length=True, truncation=True)
    y = tokenizer.encode(target, add_special_tokens=False, max_length=window_size, pad_to_max_length=True, truncation=True)
    y[:len(qa["title"]) - 1] = [0] * (len(qa["title"]) - 1)
    return x, y, len(qa["title"])


# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset(tokenizer, batch_size, window_size, corpus):
    dataset_x = []
    dataset_y = []
    dataset_q_len = []
    for i in range(batch_size):
        x, y, title_len = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
        dataset_q_len.append(title_len)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y), dataset_q_len


# 建立模型
def build_model():
    model = LanguageModel()
    return model


# 文本生成测试代码
def generate_sentence(tokenizer, openings, model, window_size):
    openings = "问："+openings + "  答："
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了[sep]，或生成文本超过100字则终止迭代
        while pred_char != "[sep]" and len(openings) <= 100:
            openings += pred_char
            x = tokenizer.encode(openings[-window_size:], add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model([], x)[0][-1]
            index = sampling_strategy(y)
            pred_char = ''.join(tokenizer.decode(index))
    return openings


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


# 计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 10  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    train_sample = 640  # 每轮训练总共训练的样本总数
    window_size = 80  # 样本文本长度
    QAs = load_QAs(corpus_path)  # 加载语料
    model = build_model()  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    tokenizer = BertTokenizer.from_pretrained(r"/home/lufei/AI/bert-base-chinese")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y, q_len = build_dataset(tokenizer, batch_size, window_size, QAs)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(q_len, x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence(tokenizer, "北京明年拟推工作日半价观看电影", model, window_size))
        print(generate_sentence(tokenizer, "韩国人用工资买房实现品质生活", model, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    train("/home/lufei/AI/week11 问答/sample_data.json", False)
