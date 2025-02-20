# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel
from transformers import BertTokenizer
import json

"""
基于pytorch的bert语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab_size, model_path, ques_max_len, ans_max_len):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_path, return_dict=False)
        self.classify = nn.Linear(input_dim, vocab_size)
        self.loss = nn.functional.cross_entropy
        self.ques_max_len = ques_max_len
        self.ans_max_len = ans_max_len

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        if y is not None:
            mask = torch.ones((x.shape[0], x.shape[1], x.shape[1]))
            for i in range(mask.shape[1]):
                if i < self.ques_max_len:
                    mask[:, i, self.ques_max_len:] = 0
                else:
                    mask[:, i, i + 1:] = 0
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            y_pred = y_pred[:, self.ques_max_len - 1:-1, :]
            return self.loss(y_pred.reshape(-1, y_pred.shape[-1]), y.reshape(-1), ignore_index=0)
        else:
            # 训练时不需要mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)


# 加载语料
def load_corpus(path):
    corpus = []
    with open(path, encoding="utf8") as f:
        for line in f:
            corpus.append(line.strip())
    return corpus


# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, ques_max_len, ans_max_len, corpus):
    index = random.randint(0, len(corpus) - 1)
    line = corpus[index]
    line = json.loads(line)
    title = line["title"]
    content = line["content"]
    title_ids = tokenizer.encode(title, add_special_tokens=False, max_length=ans_max_len, padding="max_length",
                                 truncation=True)
    content_ids = tokenizer.encode(content, add_special_tokens=False, max_length=ques_max_len, padding="max_length",
                                   truncation=True)
    x = content_ids + [tokenizer.sep_token_id] + title_ids
    y = title_ids + [tokenizer.sep_token_id]
    return x, y


# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# ques_max_len 问题长度
# ans_max_len 回答长度
# corpus 语料字符串
def build_dataset(tokenizer, sample_length, ques_max_len, ans_max_len, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(tokenizer, ques_max_len, ans_max_len, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab_size, char_dim, model_path, ques_max_len, ans_max_len):
    model = LanguageModel(char_dim, vocab_size, model_path, ques_max_len, ans_max_len)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, tokenizer, ques_max_len, ans_max_len):
    model.eval()
    x = tokenizer.encode(openings, add_special_tokens=False, max_length=ques_max_len, padding="max_length",
                         truncation=True)
    x = x + [tokenizer.sep_token_id]
    x = torch.LongTensor([x])
    with torch.no_grad():
        ans = ""
        pred_char = ""
        # 生成了SEP符，或生成文本超过ans_max_len则终止迭代
        while pred_char != "[SEP]" and len(ans) <= ans_max_len:
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = "".join(tokenizer.decode(index))
            ans += pred_char
            torch_index = torch.LongTensor([[index]])
            if torch.cuda.is_available():
                torch_index = torch_index.cuda()
            x = torch.cat((x, torch_index), dim=1)
    return openings + "---->" + ans


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
    epoch_num = 5  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    char_dim = 768  # 每个字的维度
    ques_max_len = 125  # 问题最大长度
    ans_max_len = 25  # 回答最大长度
    corpus = load_corpus(corpus_path)  # 加载语料
    model_path = r"D:\bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = build_model(tokenizer.vocab_size, char_dim, model_path, ques_max_len, ans_max_len)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(tokenizer, batch_size, ques_max_len, ans_max_len, corpus)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("阿根廷布宜诺斯艾利斯省奇尔梅斯市一服装店，8个月内被抢了三次。最后被抢劫的经历，更是直接让", model, tokenizer, ques_max_len,
                                ans_max_len))
        print(generate_sentence("中宣部昨天召开“践行雷锋精神”新闻发布会，明确指出学习雷锋精神是当前加强社会思想道德建设的需要", model, tokenizer,
                                ques_max_len, ans_max_len))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    train("sample_data.json", False)
