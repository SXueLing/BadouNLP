#coding:utf8

import json
import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel,BertTokenizer
"""
基于pytorch的LSTM语言模型
"""
def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab_size):
        super(LanguageModel, self).__init__()
        #self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        self.layer = BertModel.from_pretrained('bert-base-chinese', return_dict=False)
        hidden_size = self.layer.config.hidden_size

        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, mask=None):
        #x = self.embedding(x)       #output shape:(batch_size, sen_len, input_dim)
        #trg_mask = get_pad_mask(x, self.pid_idx) & get_subsequent_mask(x)
        if y is not None:
            x,_ = self.layer(x,mask)        #output shape:(batch_size, sen_len, input_dim)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x,_ = self.layer(x,mask)        #output shape:(batch_size, sen_len, input_dim)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)

# #加载字表
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
    # 加载题目和内容
    with open(path, encoding="utf8") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            title = line["title"]
            content = line["content"]
            corpus.append((title, content))
    return corpus

# 内容是提示，标题是答案
def build_sample(tokenizer, corpus, max_len):
    title,content = corpus
    t = tokenizer.encode(title, add_special_tokens=False)
    c = tokenizer.encode(content, add_special_tokens=False)
    x = [tokenizer.cls_token_id] + c + [tokenizer.sep_token_id] + t + [tokenizer.sep_token_id]
    y = len(c) * [-100] + [-100] + t + [tokenizer.sep_token_id] + [-100]
    mask = get_mask(len(c),len(t))

    x = x[:max_len] + [0] * (max_len - len(x))
    y = y[:max_len] + [0] * (max_len - len(y))
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)
    mask = pad_mask(mask, (max_len, max_len))
    return x, y , mask

def get_mask(c_len, t_len):
    c_len = c_len + 2
    t_len = t_len + 1
    mask = torch.ones(c_len + t_len,c_len + t_len)
    for i in range(c_len):
        mask[i, c_len:] = 0
    for i in range(t_len):
        mask[c_len + i, c_len + i + 1:] = 0
    return mask

def pad_mask(mask, shape):
    w,h = mask.shape
    new_w,new_h = shape
    padded_mask = torch.zeros(new_w,new_h)

    pw = min(w,new_w)
    ph = min(h,new_h)

    padded_mask[:pw, :ph] = mask[:pw, :ph]
    return padded_mask

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(batch, sample_length, tokenizer, corpus, max_len):
    dataset_x = []
    dataset_y = []
    dataset_mask = []
    for i in range(sample_length):
        x, y ,mask = build_sample(tokenizer, corpus[batch + i], max_len)
        dataset_x.append(x)
        dataset_y.append(y)
        dataset_mask.append(mask)
    # 将列表转换为张量
    dataset_x = torch.stack(dataset_x)  # 将列表中的张量堆叠为一个张量
    dataset_y = torch.stack(dataset_y)
    dataset_mask = torch.stack(dataset_mask)
    return dataset_x, dataset_y, dataset_mask
#建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
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


def train(corpus_path, save_weight=True):
    epoch_num = 100        #训练轮数
    batch_size = 64       #每次训练样本个数
    char_dim = 256        #每个字的维度
    vocab_size = 21128      #字表大小
    max_len = 50          #最大长度

    pretrain_model_path = r'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(vocab_size, char_dim)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(len(corpus) // batch_size):
            x, y, mask = build_dataset(batch, batch_size, tokenizer, corpus, max_len) #构建一组训练样本
            if torch.cuda.is_available():
                x, y, mask = x.cuda(), y.cuda(), mask.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y, mask)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        # print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer, window_size))
        # print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, window_size))

    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("week10/transformers-生成文章标题/sample_data.json", False)
