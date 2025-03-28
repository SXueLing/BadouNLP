#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import json
import re
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader

"""
通过bery模型以及mask形状，实现一个sft任务。
训练方式：输入问题和答案，通过正确构建lable，和预测答案做loss
"""


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)

        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)

        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, mask=None):
        if y is not None:
            #mask
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)

class DataProvider:
    def __init__(self, corpus_path, tokenizer, max_length):
        self.corpus_path = corpus_path
        self.cls = 101
        self.sep = 102
        self.max_token = max_length
        self.tokenizer = tokenizer
        self.load()

    def load(self):
        self.data = []
        print(self.corpus_path)
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                line = json.loads(line)
                question = line["title"]
                answer = line["content"]
                self.generate_data(question, answer)
        return

    def generate_data(self, question, answer):
        """
        labels生成时要把不计算loss的部分置为-1，要逐个token对齐；第一个sep（作为起始符）要对齐答案的第一个字
        """
        question_seq = self.tokenizer.encode(question, add_special_tokens=False)
        answer_seq = self.tokenizer.encode(answer, add_special_tokens=False)
        x = [self.cls] + question_seq + [self.sep] + answer_seq + [self.sep]
        y = [-1] + [-1] * len(question_seq) + answer_seq + [self.sep]
        mask = self.generate_attn_mask(len(question_seq), len(answer_seq), self.max_token)
        x = x[:self.max_token] + [0] * (self.max_token - len(x))
        y = y[:self.max_token] + [0] * (self.max_token - len(y))

        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        mask = torch.LongTensor(mask)
        self.data.append([x, y, mask])

        return

    def generate_attn_mask(self, question_len, answer_len, max_token):
        """
        1 生成attention mask matrix
        1.1 attention矩阵的长宽除了句子长度外，要计算token的个数。
        1.2 对于问题，答案完全不可见；对于答案，问题完全可见，但对于答案自身要逐字可见，要做成上三角矩阵
        2 mask矩阵也要按照max_length来填充。整个过程中，始终保持和文本的长度对应就不会出错
        """
        rows = 1 + question_len + 1  # cls + question_len + sep
        cols = answer_len + 1  # answer_len + sep
        matrix = np.zeros((rows + cols, rows + cols))
        matrix[:, :rows] = 1
        upper_triangle = np.tril(np.ones((cols, cols)))
        matrix[rows:, rows:] = upper_triangle

        #mask矩阵也要填充，和输入对应
        curr_tokens = rows + cols
        actual_tokens = min(curr_tokens, max_token)
        padding_matrix = np.zeros((max_token, max_token))
        padding_matrix[:actual_tokens, :actual_tokens] = matrix[:actual_tokens, :actual_tokens]
        return padding_matrix

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def build_dataset(tokenizer, corpus, max_length, batch_size):
    provider = DataProvider(corpus, tokenizer, max_length)
    data = DataLoader(provider, batch_size=batch_size, shuffle=True)
    return data

#建立模型
def build_model(vocab, char_dim, pretrain_model_path):
    model = LanguageModel(768, 21128, pretrain_model_path)
    return model

#文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成文本超过77字则终止迭代
        while len(openings) <= 77:
            openings += pred_char
            x = tokenizer.encode(openings, add_special_tokens=False)
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

def main(corpus_path, save_weight=True):

    pretrain_model_path = r'/home/phil7/workplace/pyFIles/ai_study/1practice/bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    bert = BertModel.from_pretrained(pretrain_model_path)

    epoch_num = 20                           #训练轮数
    batch_size = 16                          #每次训练样本个数
    char_dim = bert.config.hidden_size       #bert每个字的维度
    vocab_size = bert.config.vocab_size      #bert字表大小
    max_length = 512                         #样本文本长度

    train_data = build_dataset(tokenizer, corpus_path, max_length, batch_size)  # 建立数据集
    model = build_model(vocab_size, char_dim, pretrain_model_path)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, y, mask in train_data: #构建一组训练样本
            if torch.cuda.is_available():
                x, y, mask = x.cuda(), y.cuda(), mask.cuda(),
            optim.zero_grad()           #梯度归零
            loss = model(x, y, mask)    #计算loss
            loss.backward()             #计算梯度
            optim.step()                #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("阿根廷歹徒抢服装尺码不对拿回店里换", model, tokenizer))
        print(generate_sentence("各路足坛名人陆续抵达", model, tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    main("sample_data.json", False)
