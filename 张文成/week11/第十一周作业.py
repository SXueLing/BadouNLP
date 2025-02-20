# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel

"""

基于pytorch的LSTM语言模型,基于bert的SFT，数据集格式如下，按照/n分隔。
                                                      Q
                                                      A
                                                      
                                                      Q
                                                      A
                                                      
                                                      Q
                                                      A

"""


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path,batch_size):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        self.batch_size = batch_size
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)

        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy


    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        if y is not None:
            x_seq_len = x.shape[1]
            y_seq_len = y.shape[1] if y is not None else 0
            # # 训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
            # mask = torch.ones((x.shape[0], x.shape[1], x.shape[1]))
            #
            # mask = torch.tril(mask)
            # print("x.shape",x.shape)
            # print("y.shape",y.shape)
            # print("yuan_mask.shape",mask.shape)


            # # 构建 mask 矩阵，大小为 (x_seq_len + y_seq_len) * (x_seq_len + y_seq_len)
            mask = torch.ones((x_seq_len + y_seq_len, x_seq_len + y_seq_len))

            # 左上角是全1的矩阵
            mask[:x_seq_len, :x_seq_len] = 1

            # 右上角是全0的矩阵
            mask[:x_seq_len, x_seq_len:x_seq_len + y_seq_len] = 0

            # 左下角是全1的矩阵
            mask[x_seq_len:x_seq_len + y_seq_len, :x_seq_len] = 1

            # 右下角是下1上0的阶梯矩阵
            mask[x_seq_len:x_seq_len + y_seq_len, x_seq_len:x_seq_len + y_seq_len] = torch.tril(
                torch.ones((y_seq_len, y_seq_len)))


            mask = mask.unsqueeze(0).expand(x.shape[0], -1, -1)



            print("mask.shape",mask.shape)

            if torch.cuda.is_available():
                mask = mask.cuda()
            x_padding = torch.full((128, 10), -100)
            x_padding = x_padding.cuda()

            x, _ = self.bert(torch.cat((x, y),dim=1), attention_mask=mask)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            # print(y_pred.shape)
            y = torch.cat((x_padding, y), dim=1)
            # print(y.shape)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1), ignore_index=-100)
        else:
            # 预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)




# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="utf-8") as f:
        for line in f:
            corpus += line

    # print(corpus)
    return corpus







def build_sample(tokenizer, corpus):
    # 按照 '\n\n' 分割 QA 对
    qa_pairs = corpus.split('\n\n')
    # 随机选择一个 QA 对
    random_qa = random.choice(qa_pairs)

    # print("random_qa",random_qa)


    # 按照 '\n' 分割 Q 和 A
    q, a = random_qa.split('\n')

    # 将 Q 和 A 转换成序列
    x = tokenizer.encode(q, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)
    y = tokenizer.encode(a, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)
    # print(x)
    # print(y)
    return x, y


# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset(sample_length, tokenizer, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(tokenizer, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, pretrain_model_path,batch_size):
    model = LanguageModel(768, 21128, pretrain_model_path,batch_size)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, tokenizer):
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
            # print("generate_sentence_y",y)
            index = sampling_strategy(y)
            # print("index",index)
            mmm = tokenizer.decode(index)
            pred_char = ''.join(mmm)
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

    pretrain_model_path = r'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)  # 加载语料
    model = build_model(vocab_size, char_dim, pretrain_model_path,batch_size)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, tokenizer, corpus)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算los
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("你喜欢什么水果", model, tokenizer))
        # print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train(r"qa_pairs.txt", False)
