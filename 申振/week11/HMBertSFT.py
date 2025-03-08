import json

import torch
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

    # 当输入真实标签，返回l6oss值；无真实标签，返回预测值
    def forward(self, x, mask = None,y=None):
        if y is not None:

            if mask is not None and torch.cuda.is_available():
                mask = mask.cuda()
                x, _ = self.bert(x, attention_mask=mask)
            else:
                x, _ = self.bert(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1),ignore_index=-1)
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
    corpus = []
    with open(path, encoding="utf8") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            corpus.append(line)
    return corpus


# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, sample,max_len):

    prompt_encode = tokenizer.encode(sample["title"],add_special_tokens=False)
    answer_encode = tokenizer.encode(sample["content"],add_special_tokens=False)
    x = [tokenizer.cls_token_id] + prompt_encode + [tokenizer.sep_token_id] + answer_encode + [tokenizer.sep_token_id]
    y = len(prompt_encode) * [-1] + [-1] + answer_encode + [tokenizer.sep_token_id] + [-1]
    mask = create_mask(len(prompt_encode), len(answer_encode))

    x = x[:max_len] + [0] * (max_len - len(x))
    y = y[:max_len] + [0] * (max_len - len(y))
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)
    # 已知pad_mask 入参为 mask，target_shape
    mask = pad_mask(mask,(max_len,max_len))
    return x, mask,y


def create_mask(prompt_len, answer_len):
    total_len = 1 + prompt_len + 1 + answer_len + 1  # CLS + prompt + SEP + answer + SEP
    mask = torch.zeros((total_len, total_len), dtype=torch.bool)

    # 标题部分（CLS + prompt + SEP）全可见
    prompt_end = 1 + prompt_len + 1  # CLS位置(0) + prompt长度 + SEP位置
    mask[:prompt_end, :prompt_end] = True

    # 内容部分因果可见（可看到标题和自身前面的内容）
    for i in range(prompt_end, total_len):
        mask[i, :i + 1] = True  # 允许关注所有前面的位置

    return mask

def pad_mask(mask, target_shape):
    max_len = target_shape[0]
    orig_len = mask.size(0)
    padded_mask = torch.zeros(max_len, max_len, dtype=torch.bool)
    padded_mask[:orig_len, :orig_len] = mask
    return padded_mask

# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset(sample_length, tokenizer, corpus):
    dataset = []
    samples = np.random.choice(corpus, size=sample_length, replace=True)
    for sample in samples:
        x,mask, y = build_sample(tokenizer, sample,max_len=200)
        dataset.append([x,mask,y])
    return dataset


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
    batch_size = 16  # 每次训练样本个数
    train_sample = 100  # 每轮训练总共训练的样本总数
    char_dim = 768  # 每个字的维度
    window_size = 10  # 样本文本长度
    vocab_size = 21128  # 字表大小
    learning_rate = 0.001  # 学习率

    pretrain_model_path = r'/Users/smile/PycharmProjects/nlp/bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)  # 加载语料
    model = build_model(vocab_size, char_dim, pretrain_model_path)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            dataset = build_dataset(batch_size, tokenizer, corpus)  # 构建一组训练样本
            # 将批次数据转换为Tensor并堆叠起来
            x = torch.stack([torch.tensor(sample[0]) for sample in dataset])
            mask = torch.stack([torch.tensor(sample[1]) for sample in dataset])
            y = torch.stack([torch.tensor(sample[2]) for sample in dataset])

            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, mask, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("邓亚萍：互联网要有社会担当", model, tokenizer, window_size))
        print(generate_sentence("北美洲发现肥皂人", model, tokenizer, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train(r"sample_data.json", False)
