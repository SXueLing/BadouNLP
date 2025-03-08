# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
import json
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

"""
基于pytorch的LSTM语言模型
"""

bert_path = r"D:\桌面\资料\\week6 语言模型和预训练\\bert-base-chinese"


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # 替换LSTM
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        # 用bert代替
        # bert_path = r"D:\bert-base-chinese"
        self.bert = BertModel.from_pretrained(bert_path, return_dict=False)
        self.classify = nn.Linear(input_dim, vocab)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, mask=None, y=None):
        if y is not None:
            # bert相当于有embedding + lstm
            # x = self.embedding(x)  # output shape:(batch_size, sen_len, input_dim)
            x, _ = self.bert(x, attention_mask=mask)  # output shape:(batch_size, sen_len, input_dim)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)


# 加载字表
# def build_vocab(vocab_path):
#     vocab = {"<pad>": 0}
#     with open(vocab_path, encoding="utf8") as f:
#         for index, line in enumerate(f):
#             char = line[:-1]  # 去掉结尾换行符
#             vocab[char] = index + 1  # 留出0位给pad token
#     return vocab


# 加载语料
# def load_corpus(path):
#     corpus = ""
#     with open(path, encoding="utf8") as f:
#         for line in enumerate(f):
#             corpus += line.strip()
#     return corpus


# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, x, y):
    # start = random.randint(0, len(corpus) - 1 - window_size)
    # end = start + window_size
    # window = corpus[start:end]
    # target = corpus[start + 1:end + 1]  # 输入输出错开一位
    # print(window, target)
    # x = [vocab.get(word, vocab["<UNK>"]) for word in window]   #将字转换成序号
    # y = [vocab.get(word, vocab["<UNK>"]) for word in target]
    x = tokenizer.encode(x, padding='max_length', add_special_tokens=False, truncation=True, max_length=10)
    y = tokenizer.encode(y, padding='max_length', add_special_tokens=False, truncation=True, max_length=10)
    return x, y


# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset(tokenizer, max_length, data_path, batch_size):
    dataset = []

    with open(data_path, encoding="utf8") as f:
        for _, line in enumerate(f):
            line = json.loads(line)
            prompt = line["title"]
            answer = line["content"]
            prompt_encode, answer_encode = build_sample(tokenizer, prompt, answer)
            # label 为-1 表示不参与训练
            x = [tokenizer.cls_token_id] + prompt_encode + [tokenizer.sep_token_id] + answer_encode + [
                tokenizer.sep_token_id]
            y = [-1] + len(prompt_encode) * [-1] + answer_encode + [tokenizer.sep_token_id] + [-1]
            # padding
            x = x[:max_length] + [0] * (max_length - len(x))
            y = y[:max_length] + [0] * (max_length - len(y))
            x = torch.LongTensor(x)
            y = torch.LongTensor(y)
            mask = create_mask(len(prompt_encode), len(answer_encode))
            mask = pad_mask(mask, max_length)
            dataset.append([x, mask, y])

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def create_mask(prompt_len, answer_len):  #创建mask矩阵函数
    s1_len = prompt_len + 2  # cls + sep
    s2_len = answer_len + 1  # eos
    mask = torch.ones(s1_len + s2_len, s1_len + s2_len)
    for i in range(s1_len):
        mask[i, s1_len:] = 0  # prompt不能看到answer的token
    for i in range(s2_len):
        mask[s1_len + i, s1_len + i + 1:] = 0
    return mask


def pad_mask(mask, max_length):
    pad_mask = torch.zeros(max_length, max_length, dtype=mask.dtype, device=mask.device)
    h_start = 0
    w_start = 0
    h_end = min(mask.shape[0], max_length)
    w_end = min(mask.shape[1], max_length)
    pad_mask[h_start:h_end, w_start:w_end] = mask[:h_end - h_start, :w_end - w_start]
    return pad_mask


# 建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过20字则终止迭代
        while pred_char != "\n" and len(openings) <= 50:
            openings += pred_char
            x = tokenizer.encode(openings, padding='max_length', add_special_tokens=False, truncation=True,
                                 max_length=10)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
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


def train(data_path, save_weight=True):
    epoch_num = 200  # 训练轮数
    batch_size = 128  # 每次训练样本个数
    char_dim = 768  # 每个字的维度
    max_length = 50  # 样本文本长度
    vocab = 21128  # 建立字表
    # corpus = load_corpus(data_path)  # 加载语料
    model = build_model(vocab, char_dim)  # 建立模型
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)  # 建立优化器
    train_data = build_dataset(tokenizer, max_length, data_path, batch_size)
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        for x, mask, y in train_data:
            if torch.cuda.is_available():
                x, mask, y = x.cuda(), mask.cuda(), y.cuda()
            optim.zero_grad()
            loss = model(x, mask, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("阿根廷歹徒抢服装尺码不对拿回店里换：", model, tokenizer))
        print(generate_sentence("国际通用航空大会沈阳飞行家表演队一飞机发生坠机，伤亡不明：", model, tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(data_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("sample_data.json", False)
