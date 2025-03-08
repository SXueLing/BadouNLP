#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, BertTokenizer
import json 
from torch.utils.data import Dataset, DataLoader


"""
基于pytorch的LSTM语言模型
"""



class LanguageModel(nn.Module):
    def __init__(self, Config):
        super(LanguageModel, self).__init__()
        
        #self.embedding = nn.Embedding(len(vocab), input_dim)
        #self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        self.bert = BertModel.from_pretrained(Config["pretrain_model_path"], return_dict=False, attn_implementation="eager")
        self.classify = nn.Linear(Config["hidden_size"], Config["vocab_size"])
        #self.classify = nn.Linear(input_dim, len(vocab))
        #self.dropout = nn.Dropout(0.1)
        #self.loss = nn.functional.cross_entropy()
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, mask=None, y=None):
        #x = self.embedding(x)       #output shape:(batch_size, sen_len, input_dim)
        #x, _ = self.layer(x)        #output shape:(batch_size, sen_len, input_dim)

        if y is not None:
            #mask = (1 - torch.triu(torch.ones((x.shape[0], x.shape[1], x.shape[1])), diagonal=1))
            #mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            print(mask.shape)
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)

#加载字表
#def build_vocab(vocab_path):
#    return BertTokenizer.from_pretrained(vocab_path)
    #vocab = {"<pad>":0}
    #with open(vocab_path, encoding="utf8") as f:
    #    for index, line in enumerate(f):
    #        char = line[:-1]       #去掉结尾换行符
    #        vocab[char] = index + 1 #留出0位给pad token
    #return vocab

#加载语料
def load_corpus(path):
    #corpus = ""
    #with open(path, encoding="gbk") as f:
    #    for line in f:
    #        corpus += line.strip()
    #return corpus
    corpus = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            corpus.append([line["title"], line["content"]])
    return corpus


def encode_sentence(text, Config, tokenizer):
    return tokenizer.encode(text, padding="max_length", max_length=Config["max_length"], truncation=True, add_special_tokens=False)

def decode_sentence(text, tokenizer):
    return tokenizer.decode(text)

#补齐或截断输入的序列，使其可以在一个batch内运算
#def padding(input_id, pad_token=0):
#    input_id = input_id[:Config["max_length"]]
#    input_id += [pad_token] * (Config["max_length"] - len(input_id))
#    return input_id
    
#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(window_size, corpus, tokenizer, Config):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  #输入输出错开一位
    #print(window, target)
    #x = [vocab.get(word, vocab["<UNK>"]) for word in window]   #将字转换成序号
    #y = [vocab.get(word, vocab["<UNK>"]) for word in target]
    x = tokenizer.encode(window, padding='max_length', truncation=True, max_length=Config["max_length"], add_special_tokens=False)
    y = tokenizer.encode(target, padding='max_length', truncation=True, max_length=Config["max_length"], add_special_tokens=False)
    return x, y

#sft数据构造
#x1 x2 x3 y1 y2
#x ——>  cls x1 x2 x3 sep y1 y2 sep
#y ——>  -1  -1 -1 -1 y1  y2 sep -1 
def build_dataset(corpus, tokenizer, Config):
    #dataset_x = []
    #dataset_y = []
    #for i in range(sample_length):
    #    x, y = build_sample(window_size, corpus, tokenizer, Config)
    #    dataset_x.append(x)
    #    dataset_y.append(y)
    #return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)
    max_length = Config["max_length"]
    batch_size = Config["batch_size"]
    dataset = []
    for i, (question, answer) in enumerate(corpus):
        question_encode = tokenizer.encode(question, add_special_tokens=False)
        answer_encode = tokenizer.encode(answer, add_special_tokens=False)
        x = [tokenizer.cls_token_id] + question_encode + [tokenizer.sep_token_id] + answer_encode + [tokenizer.sep_token_id]
        y = [-1] * len(question_encode) + [-1] + answer_encode + [tokenizer.sep_token_id] + [-1]
        mask = build_mask(question_encode, answer_encode)
        x = x[:max_length] + [0] * (max_length - len(x))
        y = y[:max_length] + [0] * (max_length - len(y))
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        mask = pad_mask(mask, (max_length, max_length))
        dataset.append([x, mask, y])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def build_mask(question, answer):
    len_q = len(question) + 2 
    len_a = len(answer) + 1
    mask = torch.ones(len_q + len_a, len_q + len_a)
    for i in range(len_q):
        mask[i, len_q:] = 0
    for i in range(len_a):
        mask[len_q + i, len_q + i + 1:] = 0
    return mask

def pad_mask(input_id, target_shape):
    input_h, input_w = input_id.shape
    target_h, target_w = target_shape
    result = torch.zeros(target_shape, dtype=input_id.dtype, device=input_id.device)
    h_end = min(input_h, target_h)
    w_end = min(input_w, target_w)
    result[0:h_end, 0:w_end] = input_id[:h_end, w_end]
    return result 

#建立模型
def build_model(Config):
    #model = LanguageModel(char_dim, vocab)
    model = LanguageModel(Config)
    return model

#文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    #reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        #pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while len(openings) <= 30:
            #openings += pred_char
            #x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
            
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            #pred_char = reverse_vocab[index]
            #pred_char = ''.join(tokenizer.decode(index))
            openings.append(index)
    return tokenizer.decode(openings)

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
#def calc_perplexity(sentence, model, vocab, window_size):
#    prob = 0
#    model.eval()
#    with torch.no_grad():
#        for i in range(1, len(sentence)):
#            start = max(0, i - window_size)
#            window = sentence[start:i]
#            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
#            x = torch.LongTensor([x])
#            target = sentence[i]
#            target_index = vocab.get(target, vocab["<UNK>"])
#            if torch.cuda.is_available():
#                x = x.cuda()
#            pred_prob_distribute = model(x)[0][-1]
#            target_prob = pred_prob_distribute[target_index]
#            prob += math.log(target_prob, 10)
#    return 2 ** (prob * ( -1 / len(sentence)))


def train(corpus_path, Config, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 30       #每次训练样本个数
    train_sample = 5000   #每轮训练总共训练的样本总数
    char_dim = 256        #每个字的维度
    window_size = 10       #样本文本长度
    
    learning_rate = Config["learning_rate"]

    tokenizer = BertTokenizer.from_pretrained(Config["pretrain_model_path"])
    corpus = load_corpus(corpus_path)     #加载语料
    train_data = build_dataset(corpus, tokenizer, Config)
    model = build_model(Config)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, mask, y in train_data:
            #x, y = build_dataset(batch_size, window_size, corpus, tokenizer, Config) #构建一组训练样本
            if torch.cuda.is_available():
                x, mask, y = x.cuda(), mask.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, mask, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("北美洲发现肥皂人", model, tokenizer))
        print(generate_sentence("互联网要有社会担当", model, tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    from config import Config
    corpus_path = Config["data_path"]
    train(corpus_path, Config, False)
