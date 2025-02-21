#coding:utf8
import json

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel
from loader import load_data

# from 文本分类.use import padding

"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path, content_max_len):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)

        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)

        self.classify = nn.Linear(hidden_size, vocab_size)
        self.content_max_len = content_max_len
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        if y is not None:
            mask = torch.ones((x.shape[0], x.shape[1], x.shape[1]))
            for i in range(mask.shape[1]):
                if i < self.content_max_len:
                    mask[:, i, self.content_max_len:] = 0
                else:
                    mask[:, i, i + 1:] = 0
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            sep_index = self.content_max_len
            y_pred_answer = y_pred[:, sep_index:-1, :]
            return self.loss(y_pred_answer.reshape(-1, y_pred_answer.shape[-1]), y.reshape(-1), ignore_index=0)
        else:
            #预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)   #output shape:(batch_size, sen_len, vocab_size)
            return torch.softmax(y_pred, dim=-1)

#建立模型
def build_model(pretrain_model_path, content_max_len):
    model = LanguageModel(768, 21128, pretrain_model_path, content_max_len)
    return model

#文本生成测试代码
def generate_sentence(sentence, model, tokenizer, content_max_len, title_max_len=30):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    input_ids = tokenizer.encode(sentence, add_special_tokens=False, padding="max_length", max_length=content_max_len, truncation=True)
    input_ids = input_ids + [tokenizer.sep_token_id]
    input_ids = torch.LongTensor([input_ids])
    pred_title = ''
    pred_char = ''
    with torch.no_grad():
        while True:
            y = model(input_ids)[0][-1]
            index = sampling_strategy(y)
            pred_char = tokenizer.decode([index])
            if pred_char == "[SEP]":
                break
            pred_title += pred_char
            pred_id = torch.LongTensor([[index]])
            input_ids = torch.cat((input_ids, pred_id), dim=1)
            print(pred_char, end="")

    print()
    # return pred_title

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



def train(save_weight=False):
    epoch_num = 50        #训练轮数
    batch_size = 25       #每次训练样本个数
    content_max_len = 120
    title_max_len = 30
    learning_rate = 0.001  #学习率
    sample_path = "sample_data.json"

    pretrain_model_path = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    train_data  = load_data(batch_size, sample_path, tokenizer, content_max_len, title_max_len)
    model = build_model(pretrain_model_path, content_max_len)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for index, batch_data in enumerate(train_data):
            input_ids = batch_data
            optim.zero_grad()    #梯度归零
            loss = model(input_ids, input_ids[:, content_max_len + 1:])   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        generate_sentence("5月12日凌晨，他打开防盗门，给了雇来的杀手一把菜刀和一把水果刀，15分钟后，他的父亲和姐姐高玮艺倒在血泊中。他的动机简单到无法令办案警察相信：家里对他管得严，抱得希望太高，让他有些压力。 -新京报 ", model, tokenizer, content_max_len)
        generate_sentence("昨天，南京张先生在一家自助银行取款后匆忙离开，随后就连续收到取款提示短信，提示银行卡被人提款。返回提款机，只见一男青年正在操作，取出了一大叠百元钞票。取款男子自称是用取钱的方式提醒失主，算不算盗窃南京警方仍在调查。 ", model, tokenizer, content_max_len)
    if not save_weight:
        return
    else:
        torch.save(model.state_dict(), "model_mask.pth")
        return

def evaluate():
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    content_max_len = 120
    pretrain_model_path = 'bert-base-chinese'
    model = build_model(pretrain_model_path, content_max_len)    #建立模型
    model.load_state_dict(torch.load("model_mask.pth"))
    generate_sentence(
        "6月合格境外机构投资者(QFII)加快入市步伐。据中登公司发布的2013年6月份统计月报显示，QFII基金6月份在沪深两市分别新增开户14、15个A股股票账户，这29个账户让QFII在沪深两市的总账户数达到465个。 ",
        model, tokenizer, content_max_len)

if __name__ == "__main__":
    # train(True)
    evaluate()
