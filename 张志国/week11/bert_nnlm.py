#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel
import json

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
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, mask=None):
        if y is not None:
            #训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
            if mask is not None:
                if torch.cuda.is_available():
                    mask = mask.cuda()
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            #预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)

#加载字表
# def build_vocab(vocab_path):
#     vocab = {"<pad>":0}
#     with open(vocab_path, encoding="utf8") as f:
#         for index, line in enumerate(f):
#             char = line[:-1]       #去掉结尾换行符
#             vocab[char] = index + 1 #留出0位给pad token
#     return vocab

#加载语料
def load_corpus(path):
    corpus = []
    max_title_length = 0
    max_content_length = 0
    with open(path, encoding="utf8") as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                title = line["title"]
                content = line["content"]
                if len(title) > max_title_length:
                    max_title_length = len(title)
                if len(content) > max_content_length:
                    max_content_length = len(content)
                corpus.append((title, content))
    return corpus, max_title_length, max_content_length

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  #输入输出错开一位

    x = tokenizer.encode(window, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)   #将字转换成序号
    y = tokenizer.encode(target, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)

    return x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(data, tokenizer, max_title_length, max_content_length):
    dataset_x = []
    dataset_y = []
    masks = []
    for d in data:
        title = d[0]
        content = d[1]
        max_length =  max_title_length + max_content_length
        x_ = title + "[SEP]" + content
        y_ = title[1:]+ "[SEP]" + content + "[SEP]"
        x = tokenizer.encode(x_, add_special_tokens=False, padding='max_length', truncation=True, max_length=max_length)   #将字转换成序号
        y = tokenizer.encode(y_, add_special_tokens=False, padding='max_length', truncation=True, max_length=max_length)
        y = [-100 if i < len(title) or x[i] == 0 else y[i] for i in range(len(y))] #将title部分置为-100，不参与计算loss
        mask_1 = torch.ones(len(x), len(title) + 1)
        mask_2 = torch.zeros(len(title) + 1, max_length - len(title) - 1)
        mask_3 = torch.tril(torch.ones(max_length - len(title) - 1, max_length - len(title) - 1))
        mask = torch.cat((mask_2, mask_3), dim=0)
        mask = torch.cat((mask_1, mask), dim=1)
        dataset_x.append(x)
        dataset_y.append(y)
        masks.append(mask)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y), torch.stack(masks, dim=0)

#建立模型
def build_model(vocab, char_dim, pretrain_model_path):
    model = LanguageModel(768, 21128, pretrain_model_path)
    return model

#文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        # pred_char != "\n" and
        while  len(openings) <= 30:
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
    # return int(torch.argmax(prob_distribution))
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
    epoch_num = 100        #训练轮数
    batch_size = 50       #每次训练样本个数
    char_dim = 768        #每个字的维度
    vocab_size = 21128      #字表大小
    learning_rate = 0.001  #学习率
    

    pretrain_model_path = r'/Users/zzg/Documents/AI/NLP/week6/bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    corpus, max_title_length, max_content_length = load_corpus(corpus_path)     #加载语料
    train_sample = len(corpus)   #每轮训练总共训练的样本总数
    dataset = build_dataset(corpus, tokenizer, max_title_length, max_content_length) #构建训练集
    model = build_model(vocab_size, char_dim, pretrain_model_path)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = dataset[0][batch * batch_size: (batch + 1) * batch_size], dataset[1][batch * batch_size: (batch + 1) * batch_size]
            mask = dataset[2][batch * batch_size: (batch + 1) * batch_size]
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y, mask)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("罗伯斯干扰刘翔是否蓄谋已久", model, tokenizer))
        print(generate_sentence("APEC期间廊坊放假方案消息属实", model, tokenizer))
        
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("./sample_data.json", False)