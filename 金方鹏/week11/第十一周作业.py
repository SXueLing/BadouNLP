# coding:utf8

import torch
import json
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel

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

    def create_attention_mask(self, title_lengths, seq_length):
        """
        构造问答任务中的自定义 attention mask。
        title_lengths: 每个样本的 title 部分长度（包含 [CLS] 和第一个 [SEP]）

        规则：
        1. title 部分可以相互注意；
        2. content 部分只能关注 title 部分和自己之前的部分（自回归）。
        """
        batch_size = title_lengths.size(0)
        masks = []
        for i in range(batch_size):
            t_len = title_lengths[i].item()
            mask = torch.zeros(seq_length, seq_length)

            # 对于 title 部分：仅允许 j < t_len
            mask[:t_len, :t_len] = 1.0  # title 部分可以相互注意

            # 对于 content 部分：允许关注 title 部分及其自身之前的部分（自回归）
            for pos in range(t_len, seq_length):
                mask[pos, :t_len] = 1.0  # content 部分可以关注 title 部分
                mask[pos, pos:] = 1.0  # content 部分可以关注自己之前的部分

            masks.append(mask)

        masks = torch.stack(masks, dim=0)  # (batch_size, seq_length, seq_length)
        return masks
    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        if y is not None:
            # 训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
            mask = self.create_attention_mask(x.size(1),y.size(1))
            #print(mask)
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # 预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)


# 加载字表
# def build_vocab(vocab_path):
#     vocab = {"<pad>":0}
#     with open(vocab_path, encoding="utf8") as f:
#         for index, line in enumerate(f):
#             char = line[:-1]       #去掉结尾换行符
#             vocab[char] = index + 1 #留出0位给pad token
#     return vocab

# 加载语料
def load_corpus(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append({
                "title": item["title"],
                "context": item["content"]
            })
    return data


# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, item):
    # start = random.randint(0, len(corpus) - 1 - window_size)
    # end = start + window_size
    # window = corpus[start:end]
    # target = corpus[start + 1:end + 1]  # 输入输出错开一位
    title = item["title"]
    context = item["context"]

    x = tokenizer.encode(title, add_special_tokens=False, padding='max_length', truncation=True, max_length=32)  # 将字转换成序号
    y = tokenizer.encode(context, add_special_tokens=False, padding='max_length', truncation=True, max_length=64)
    # print(x,y)
    return x, y


# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset(tokenizer, corpus):
    dataset_x = []
    dataset_y = []
    #data = load_corpus(r'../transformers-生成文章标题/sample_data.json')
    #i=0
    for item in corpus:
        #i+=1
        #print(item, i)
        x, y = build_sample(tokenizer, item)
        # print(x,y)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, pretrain_model_path):
    model = LanguageModel(768, 21128, pretrain_model_path)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, tokenizer, window_size):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过30字则终止迭代
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
    epoch_num = 20  # 训练轮数
    batch_size = 128  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    char_dim = 768  # 每个字的维度
    window_size = 10  # 样本文本长度
    vocab_size = 21128  # 字表大小
    learning_rate = 0.001  # 学习率

    pretrain_model_path = r"E:\AI\第六周 语言模型\bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)  # 加载语料
    # print(corpus)
    build_dataset(tokenizer, corpus)

    model = build_model(vocab_size, char_dim, pretrain_model_path)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(tokenizer, corpus)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("中消协教你选面膜 贵的不代表好", model, tokenizer, window_size))
        print(generate_sentence("阿根廷歹徒抢服装尺码不对拿回店里换", model, tokenizer, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train(r"../transformers-生成文章标题/sample_data.json", False)
