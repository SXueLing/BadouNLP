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
        self.layer = BertModel.from_pretrained(r"..\..\第六周 语言模型\bert-base-chinese", return_dict=False)
        self.classify = nn.Linear(self.layer.config.hidden_size, 21128)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, q_len, x, y=None):
        mask = []
        for sen, qlen in zip(x, q_len):
            matrix = [[1 for _ in range(len(sen))] for _ in range(len(sen))]
            for i in range(qlen):
                for j in range(qlen, len(sen)):
                    matrix[i][j] = 0
            for i in range(qlen, len(sen)):
                k = 1
                for j in range(qlen + k, len(sen)):
                    k = k + 1
                    matrix[i][j] = 0
            mask.append(matrix)
        # print(mask)
        mask = torch.LongTensor(mask)

        if torch.cuda.is_available():
            mask = mask.cuda()
        if y is not None:
            x, _ = self.layer(x, attention_mask=mask)  # output shape:(batch_size, sen_len, hidden_size)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.layer(x)  # output shape:(batch_size, sen_len, hidden_size)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)


def generate_upper_triangular_mask(size, batch_size):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


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
    target = "<S>" + window[:len(window) - 2]
    target = target + "[SEP]"
    x = tokenizer.encode(window, add_special_tokens=False, max_length=window_size, pad_to_max_length=True)
    y = tokenizer.encode(target, add_special_tokens=False, max_length=window_size, pad_to_max_length=True)
    y[:len(qa["title"]) - 1] = [-100] * (len(qa["title"]) - 1)
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
    window_size = 120  # 样本文本长度
    QAs = load_QAs(corpus_path)  # 加载语料
    model = build_model()  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    tokenizer = BertTokenizer.from_pretrained(r"..\..\第六周 语言模型\bert-base-chinese")
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
        print(generate_sentence(tokenizer, "两人受伤一厂锅炉发生爆炸", model, window_size))
        print(generate_sentence(tokenizer, "中宣部推动学雷锋 进教材开微博", model, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    train("sample_data.json", False)
