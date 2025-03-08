# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import os
from transformers import BertTokenizer, BertModel
import json


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy  # ignore_index

    def forward(self, x, y=None):
        max_title_length = 20
        max_content_length = 100
        total_length = max_title_length + max_content_length
        mask = torch.ones((x.shape[0], total_length, total_length), device=x.device)
        mask[:, :max_title_length, max_title_length:] = 0
        mask[:, max_title_length:, max_title_length:] = torch.tril(
            torch.ones(max_content_length, max_content_length, device=x.device))

        if y is not None:
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)


# 加载语料
def load_corpus(path):
    corpus = []
    with open(path, encoding="utf-8") as f:
        for line in f:  # 逐行读取
            # 去除可能的空白字符
            cleaned_line = line.strip()
            if cleaned_line:
                corpus.append(json.loads(cleaned_line))
    return corpus

def build_sample(tokenizer, window_size, corpus):
    max_title_length = 20
    max_content_length = 100
    sample = random.choice(corpus)
    title_ids = tokenizer.encode(
        sample['title'],
        max_length=max_title_length,
        truncation=True,
        add_special_tokens=False
    )[:max_title_length]  # 确保不超长
    title_padded = title_ids + [tokenizer.pad_token_id] * (max_title_length - len(title_ids))
    content_ids = tokenizer.encode(
        sample['content'],
        max_length=max_content_length,
        truncation=True,
        add_special_tokens=False
    )
    input_ids = title_padded + content_ids[:-1]
    labels = [-100] * max_title_length + content_ids[1:]

    # 填充到总长度（title+content部分的总长度固定）
    total_length = max_title_length + max_content_length
    input_ids += [tokenizer.pad_token_id] * (total_length - len(input_ids))
    labels += [-100] * (total_length - len(labels))

    return (input_ids, labels)

def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(tokenizer, window_size, corpus)
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
        while pred_char != "\n" and len(openings) <= 120:
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

    pretrain_model_path = r"D:\projects\nlp\week6\bert-base-chinese\bert-base-chinese"
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
            x, y = build_dataset(batch_size, tokenizer, window_size, corpus)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    train(
        "D:/projects/nlp/week11/sample_data.json",
        False)
