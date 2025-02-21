#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from tqdm import tqdm
from transformers import BertTokenizer,BertModel
from torch.utils.data import Dataset, DataLoader
import json
"""
基于pytorch的bert语言模型增加STF
"""


class LanguageModel(nn.Module):
    def __init__(self, pretrain_model_path):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')
        hidden_size = self.bert.config.hidden_size
        vocab_size = self.bert.config.vocab_size
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, mask=None):
       if y is not None:
           if torch.cuda.is_available():
               mask = mask.cuda()
           x, _ = self.bert(x,attention_mask = mask)
           y_pred = self.classify(x)
           return self.loss(y_pred.view(-1,y_pred.shape[-1]),y.view(-1))
       else:
           x, _ = self.bert(x)
           y_pred = self.classify(x)
           return torch.softmax(y_pred, dim=-1)

#建立数据集
class Data(Dataset):
    def __init__(self,data_path,tokenizer,sentence_len):
        self.tokenizer = tokenizer
        self.sentence_len = sentence_len
        self.data = []
        with open(data_path, encoding='utf8') as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                title, content = line['title'], line['content']
                self.prepare_data(title, content)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def prepare_data(self, title, content):
        #start = random.randint(0, max(0,len(content) - 1 - self.sentence_len))
        #end = min(start + self.sentence_len,len(content))
        #x = title + content[start:end]
        #y = title + content[start + 1:end + 1]#输入输出错开一位
        x = title + '[SEP]' + content
        y = title[1:] + '[SEP]' + content + '[EOS]'
        x = self.tokenizer.encode(
            x, add_special_tokens=False, padding='max_length',
            truncation=True, max_length=self.sentence_len
        )
        y = self.tokenizer.encode(
            y, add_special_tokens=False, padding='max_length',
            truncation=True, max_length=self.sentence_len
        )
        y[:len(title)] = [-100] * len(title)
        mask = torch.tril(torch.ones(self.sentence_len, self.sentence_len))
        mask[:len(title)+1, :len(title)+1] = 0.0
        self.data.append([torch.LongTensor(x), torch.LongTensor(y), mask])
        return

def load_data(data_path, tokenizer, sentence_len, batch_size, shuffle=True):
    dataset = Data(data_path, tokenizer, sentence_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

#建立模型
def build_model(pretrain_model_path):
    model = LanguageModel(pretrain_model_path)
    return model

#文本生成测试代码
def generate_sentence(openings, model, tokenizer, sentence_len):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != '[EOS]' and len(openings) <= sentence_len:
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

def train(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 16       #每次训练样本个数
    sentence_len = 128       #样本文本长度
    pretrain_model_path = r"D:\NLP\bert-base-chinese\bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    train_data = load_data(corpus_path, tokenizer, sentence_len, batch_size, True)  # 加载语料
    model = build_model(pretrain_model_path)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for _, (x, y, mask) in tqdm(enumerate(train_data), total=len(train_data)):
            if torch.cuda.is_available():
                x, y, mask = x.cuda(), y.cuda(), mask.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y, mask)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("我们这一代：人生第一笔工资，你怎么花的？", model, tokenizer, sentence_len))
        print(generate_sentence("罗伯斯干扰刘翔是否蓄谋已久？", model, tokenizer, sentence_len))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return

if __name__ == "__main__":
    train("sample_data.json")
