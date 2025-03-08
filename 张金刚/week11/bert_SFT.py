#coding:utf8
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
基于pytorch的LSTM语言模型
"""
use_bert=True
bert_path=r"D:\myPython\Time-LLM-main\bert-base-chinese"
bert_vocab_path=r"D:\myPython\Time-LLM-main\bert-base-chinese\vocab.txt"
corpus_path="corpus.txt"
# corpus_path="/mnt/workspace/corpus.txt"
# bert_path=r"/mnt/workspace/bert-base-chinese"
# bert_vocab_path=r"/mnt/workspace/bert-base-chinese/vocab.txt"
class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab,title_len,contents_len):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), input_dim)
        self.encoder = BertModel.from_pretrained(bert_path, return_dict=False)
        self.classify = nn.Linear(input_dim, len(vocab))
        self.dropout = nn.Dropout(0.1)
        self.title_len=title_len
        self.contents_len=contents_len

        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self,x, y=None):

        if y is not None:
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            # 将左上方部分置为1
            mask[:, :self.title_len, :self.title_len] = 1
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.encoder(x, attention_mask=mask)

            x=x[:, -self.contents_len:, :] # 截取每个样本的最后 生成内容长度的 个时间步
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)

            y_pred= y_pred.view(-1, y_pred.shape[-1])   #[batch_size * seq_length, vocab_size]
            y= y.view(-1)   #[batch_size * seq_length]
            return self.loss(y_pred, y)
        else:
            x, _ = self.encoder(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)

#加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>":0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index + 1 #留出0位给pad token
    return vocab
#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample( window_size,title_len,corpus):

    tokenizer = BertTokenizer.from_pretrained(bert_path, return_dict=False)
    data=random.choice(corpus)
    # print(data)

    title = data["title"]
    content = data["content"][:window_size]# 截取前window_size个字

    x = tokenizer.encode(title, add_special_tokens=False, truncation=True, max_length=title_len,
                                 pad_to_max_length=True)
    y = tokenizer.encode(content, add_special_tokens=False, truncation=True, max_length=window_size,
                                 pad_to_max_length=True)
            # self.prepare_data(title, content)

    x=( x+y)[:-1]#输入输出错开一位
    # print(window, target)
    return x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#title_len 标题长度
#corpus 语料字符串
def build_dataset(sample_length, window_size,title_len,corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample( window_size,title_len,corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim,title_len,contents_len):
    model = LanguageModel(char_dim, vocab,title_len,contents_len)
    return model

#文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            if use_bert:
                tokenizer = BertTokenizer.from_pretrained(bert_path, return_dict=False)
                x = tokenizer.encode(openings, add_special_tokens=False, truncation=True, max_length=window_size, pad_to_max_length=True)
            else :
                x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = reverse_vocab[index]
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


#计算文本ppl
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
    return 2 ** (prob * ( -1 / len(sentence)))
def load_corpus(path):
    corpus = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))
    return corpus

def train(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 64       #每次训练样本个数
    train_sample = 500  #每轮训练总共训练的样本总数
    char_dim = 768        #每个字的维度
    window_size = 10       #样本文本长度，等于生成文本内容长度
    contents_len=window_size
    title_len = 10         #标题长度
    vocab = build_vocab(bert_vocab_path)  # 建立字表
    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(vocab, char_dim,title_len,contents_len)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):

            x, y = build_dataset(batch_size, window_size,title_len,corpus) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("中国八成家庭油盐摄入量超标 慢性病负担增长将超80%", model, vocab, window_size))
        print(generate_sentence("潜伏者涂兆兴：敌人眼皮下掩护红色后代", model, vocab, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train(r"..\transformers-生成文章标题\sample_data.json", False)
