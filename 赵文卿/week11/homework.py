import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel
import logging
import json

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LanguageModel(nn.Module):
    def __init__(self, char_dim, vocab_size, pretrain_model_path, title_len, contents_len):
        super(LanguageModel, self).__init__()
        self.title_len=title_len
        self.contents_len=contents_len

        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
        self.classify = nn.Linear(char_dim, len(vocab_size))
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        if y is not None:
            batch_size, seq_len = x.shape

            # 构建Attention Mask
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            mask[:, :self.title_len, :self.title_len] = 1
                
            if torch.cuda.is_available():
                mask = mask.cuda()            
            x, _ = self.bert(x, attention_mask=mask)
            x = x[:, -self.contents_len:, :]
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1), ignore_index=-100)
        else:
            # 预测时使用全注意力（需自行实现生成逻辑）
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)
#加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>":0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index + 1 #留出0位给pad token
    return vocab

#加载语料
def load_corpus(path):
    corpus = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))
    return corpus

#随机生成一个样本
def build_sample(tokenizer, window_size,title_len,corpus):

    data=random.choice(corpus)

    title = data["title"]
    content = data["content"][:window_size]# 截取前window_size个字

    x = tokenizer.encode(title, add_special_tokens=False, truncation=True, max_length=title_len, padding='max_length')

    y = tokenizer.encode(content, add_special_tokens=False, truncation=True, max_length=window_size,padding='max_length')

    x = x + y
    x = x[:-1]
    return x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, tokenizer,window_size, title_len, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(tokenizer, window_size,title_len, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab_size, char_dim, pretrain_model_path, title_len, contents_len):
    model = LanguageModel(char_dim, vocab_size,pretrain_model_path, title_len, contents_len)
    return model

#文本生成测试代码
def generate_sentence(openings, model, tokenizer, window_size):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
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
    batch_size = 64       #每次训练样本个数
    train_sample = 10000   #每轮训练总共训练的样本总数
    char_dim = 768        #每个字的维度
    window_size = 10        #样本文本长度，等于生成文本内容长度
    contents_len = 10   
    title_len = 10
    pretrain_model_path = r'E:\bert-base-chinese'
    vocab_path = r'E:\bert-base-chinese\vocab.txt'
    vocab_size = build_vocab(vocab_path)      #字表大小
    corpus = load_corpus(corpus_path) #加载语料
    learning_rate = 1e-5    #学习率
    
    #建立模型
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    model = build_model(vocab_size, char_dim,pretrain_model_path, title_len, contents_len)    
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    logger.info("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, tokenizer, window_size, title_len, corpus)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        logger.info("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        logger.info(generate_sentence("晚安魔都夜景", model, tokenizer, window_size))
        logger.info(generate_sentence("美副总统称中国不能自由呼吸 中国学生要求其道歉", model, tokenizer, window_size))
    #return
    if not save_weight:
        return
    else:
        torch.save(model.state_dict(), "week11/model/nnlm.pth")
        logger.info("模型已保存")
        return

if __name__ == "__main__":
    train(r"week10/data/sample_data.json", False)
