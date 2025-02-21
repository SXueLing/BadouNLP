#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel

"""
自回归语言模型训练
 bert+mask变为自回归语言模型训练
"""


class LanguageModel(nn.Module):
    # def __init__(self, input_dim, vocab, pretrain_model_path):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)

        # attn_implemention='eager'是因为旧版本的transformers的mask的shape为(batch_size, seq_len, seq_len)
        # 新版本的transformers的mask的shape为(batch_size, seq_len)
        # self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implemention='eager')
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)

        # self.classify = nn.Linear(input_dim, len(vocab))

        # linear层的输入是bert的hidden_size，输出是vocab_size
        # 是因为bert的输出是hidden_size维的向量，需要映射到vocab_size维的向量，作为每个词的概率分布
        self.classify = nn.Linear(hidden_size, vocab_size)

        # self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    # def forward(self, x, y=None):
    #     x = self.embedding(x)       #output shape:(batch_size, sen_len, input_dim)
    #     x, _ = self.layer(x)        #output shape:(batch_size, sen_len, input_dim)
    #     y_pred = self.classify(x)   #output shape:(batch_size, sen_len, vocab_size)
    #     if y is not None:
    #         return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
    #     else:
    #         return torch.softmax(y_pred, dim=-1)
    def forward(self, x, y=None):
        if y is not None: # 有标签，训练模式
            # 构建mask, 用于屏蔽无用的token
            # 注意新旧版本的transformers的mask的shape不一样,新版本的transformers的mask的shape为(batch_size, seq_len),旧版本的transformers的mask的shape为(batch_size, seq_len, seq_len)
            # 如果使用旧版本的transformers，需要加一个参数进行说明:attn_implemention='eager'
            # 源码中attn_implemention='sdpa'，这个参数会决定use_sdpa_attention_masks的值，如果为True，则mask的shape为(batch_size, seq_len)，如果为False，则mask的shape为(batch_size, seq_len, seq_len)
            # print("y.shape:", y.shape)   # y.shape:（batch_size, seq_len）
            # print("x.shape1:", x.shape)  # x.shape1:（batch_size, seq_len）
            mask = torch.tril(torch.ones(x.shape[0],x.shape[1], x.shape[1]))  # 生成下三角矩阵
            if torch.cuda.is_available():  # 如果有cuda
                mask = mask.cuda()  # 将mask放到cuda上
            x, _ = self.bert(x, attention_mask=mask)  
            # print("x.shape2:", x.shape)  # x.shape2:（batch_size, seq_len, hidden_size）
            y_pred = self.classify(x)   # output shape:(batch_size, vocab_size)
            # print("y_pred.shape:", y_pred.shape)
            # print("y.shape:", y.shape)
            # 交叉熵的参数shape是固定的，(N, C)和(N,)
            # input：(N, C)形状的张量，其中N为Batch_size，C为类别数。该参数对应于神经网络最后一个全连接层输出的未经Softmax处理的结果。
            # target：一个大小为(N,)张量，其值是0 <= targets[i] <= C-1的元素，其中C是类别数。该参数包含一组给定的真实标签（ground truth）。
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:  # 无标签，预测模式
            x, _ = self.bert(x)
            y_pred = self.classify(x)   # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)  # dim=-1表示对最后一个维度进行softmax操作，即对每个词的概率分布进行softmax操作

# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample_bert(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)  # 随机生成一个起始位置
    end = start + window_size   # 结束位置
    window = corpus[start:end]  # 截取窗口
    target = corpus[start + 1:end + 1]  # 输入输出错开一位，看ppt中自回归模型训练的图
    # print(window, target)
    # x = [vocab.get(word, vocab["<UNK>"]) for word in window]   # 将字转换成序号
    # y = [vocab.get(word, vocab["<UNK>"]) for word in target]
    x = tokenizer.encode(window, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)   # 将字转换成序号
    y = tokenizer.encode(target, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)
    return x, y


# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset_bert(sample_length, tokenizer, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        # x, y = build_sample(vocab, window_size, corpus)
        x, y = build_sample_bert(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model_bert(hidden_size, vocab_size,pretrain_model_path):
    # 768是bert的hidden_size，21128是bert的vocab_size，可以从config.json中查看
    model = LanguageModel(hidden_size, vocab_size, pretrain_model_path)
    return model


# 文本生成测试代码
def generate_sentence_bert(openings, model, tokenizer, window_size):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())  # 将词表反转，方便根据下标找到对应的字
    model.eval()  # 模型进入测试模式
    with torch.no_grad():  # 不需要计算梯度
        pred_char = ""
        # 生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            # x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
            x = tokenizer.encode(openings, add_special_tokens=False)  # 将字转换成序号
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]  # x送入模型，[0]是因为这个batch内只有一个样本，[-1]是取最后一个预测结果（即下一个字的概率分布）
            index = sampling_strategy(y)
            # pred_char = reverse_vocab[index]  # 根据下标找到对应的字
            pred_char = ''.join(tokenizer.decode(index))
    return openings

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":  # 贪心策略
        return int(torch.argmax(prob_distribution))  # 返回概率最大的下标
    elif strategy == "sampling":  # 采样策略
        prob_distribution = prob_distribution.cpu().numpy() # 转换成numpy数组
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)  # 根据概率分布进行采样


def train(corpus_path, save_weight=True):
    epoch_num = 20        # 训练轮数
    batch_size = 64       # 每次训练样本个数
    train_sample = 10000   # 每轮训练总共训练的样本总数
    char_dim = 768        # 每个字的维度
    window_size = 10       # 样本文本长度
    vocab_size = 21128     # 字表大小
    learning_rate = 0.001  # 学习率

    # vocab = build_vocab("vocab.txt")       # 建立字表

    # 用bert模型，得用bert自带的tokenizer，不需要自己建立字表
    pretrain_model_path = r'./badou/week6/bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)  # 加载bert分词器，用于分词

    corpus = load_corpus(corpus_path)     # 加载语料

    # model = build_model(vocab, char_dim)    # 建立模型
    model = build_model_bert(char_dim, vocab_size,pretrain_model_path)  # 建立模型

    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    import time
    print("start time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            # x, y = build_dataset(batch_size, vocab, window_size, corpus) #构建一组训练样本
            x, y = build_dataset_bert(batch_size, tokenizer, window_size, corpus) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print("第"+ str(epoch + 1) +"轮end time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # openings引言，即生成的文本的开头
        # print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        # print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
        print(generate_sentence_bert("让他在半年之前，就不能做出", model, tokenizer, window_size))
        print(generate_sentence_bert("李慕站在山路上，深深的呼吸", model, tokenizer, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train(r"./badou/week10 文本生成问题/lstm语言模型生成文本/corpus.txt", False)
