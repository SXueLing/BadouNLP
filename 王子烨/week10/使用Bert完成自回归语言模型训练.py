# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os

from transformers import BertModel, BertTokenizer

"""
基于 BERT 的语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(r'E:\model\bert-base-chinese', return_dict=False)
        self.classify = nn.Linear(768, vocab_size)  # 分类头，输出维度 = 词表大小
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        if y is not None:
            mask = torch.tril(torch.ones(x.shape[0],x.shape[1],x.shape[1]))
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.bert(x,attention_mask = mask)
            y_pred = self.classify(x)  # (batch_size, seq_len, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))  # 计算 loss
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # (batch_size, seq_len, vocab_size)
            return torch.softmax(y_pred, dim=-1)  # 返回概率分布


# 加载 BERT 词表
def build_vocab(bert_path):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    vocab = tokenizer.get_vocab()  # 获取 BERT 词表
    return vocab, tokenizer


# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


# 生成一个样本
def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位

    x = tokenizer.encode(window, padding='max_length', truncation=True,
                         max_length=10)  # 将字转换成序号
    y = tokenizer.encode(target, padding='max_length', truncation=True, max_length=10)

    return x, y


# 构建数据集
def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_x, dataset_y = [], []
    for _ in range(sample_length):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 构建模型
def build_model(vocab_size):
    model = LanguageModel(vocab_size)
    return model


# 文本生成
def generate_sentence(openings, model, tokenizer, window_size):
    model.eval()
    with torch.no_grad():
        pred_token = ""
        while pred_token != "[SEP]" and len(openings) <= 30:
            tokens = tokenizer(openings, add_special_tokens=False, return_tensors="pt", padding="max_length",
                               truncation=True,
                               max_length=window_size)
            x = tokens["input_ids"]

            if torch.cuda.is_available():
                x = x.cuda()

            y = model(x)[0][-1]  # 取最后一个 token 的 logits
            index = sampling_strategy(y)  # 采样一个 token ID
            pred_token = tokenizer.decode([index])  # 反解成文本

            openings += pred_token

    return openings


# 采样策略
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


# 计算困惑度 (Perplexity)
def calc_perplexity(sentence, model, tokenizer, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]

            tokens = tokenizer(window, return_tensors="pt", padding="max_length", truncation=True,
                               max_length=window_size)
            x = tokens["input_ids"]

            target = sentence[i]
            target_tokens = tokenizer(target, return_tensors="pt")["input_ids"][0]

            if torch.cuda.is_available():
                x = x.cuda()

            pred_prob_distribute = model(x)[0, -1]  # 取最后一个 token 的概率分布
            target_prob = pred_prob_distribute[target_tokens]
            prob += math.log(target_prob.item(), 10)

    return 2 ** (-prob / len(sentence))


# 训练函数
def train(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 50000  # 训练样本总数
    window_size = 10  # 样本文本长度

    vocab, tokenizer = build_vocab(r"E:\model\bert-base-chinese")  # 词表
    corpus = load_corpus(corpus_path)  # 语料
    model = build_model(len(vocab))  # 模型

    if torch.cuda.is_available():
        model = model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=0.00001)

    print("文本词表模型加载完毕，开始训练")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, tokenizer, window_size, corpus)  # 构建一组训练样本

            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()

            watch_loss.append(loss.item())

        print("=========\n第%d轮平均 loss: %f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, window_size))

    if save_weight:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    train("corpus.txt", False)
