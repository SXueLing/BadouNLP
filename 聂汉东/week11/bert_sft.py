import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel
import json

bert_path = '../bert-base-chinese'

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path, return_dict=False, attn_implementation='eager')
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        max_title_length = 20
        max_content_length = 100
        total_length = max_title_length + max_content_length
        # 创建一个全为1的mask矩阵
        mask = torch.ones(x.shape[0], total_length, total_length, device=x.device)
        # 右上角 20x100 设置为0
        mask[:, :max_title_length, max_title_length:] = 0

        # 右下角 100x100 设置为下三角矩阵
        mask[:, max_title_length:, max_title_length:] = torch.tril(
            torch.ones(max_content_length, max_content_length, device=x.device))

        if y is not None:
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)


# 加载语料
def load_corpus(corpus_path):
    corpus = []
    with open(corpus_path, encoding='utf-8') as f:
        for line in f:
            cleaned_line = line.strip()
            if cleaned_line:
                corpus.append(json.loads(cleaned_line))
    return corpus


# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, window_size, corpus):
    max_title_length = 20
    max_content_length = 100
    sample = random.choice(corpus)

    # 编码标题并截断/填充
    title_ids = tokenizer.encode(
        sample['title'],
        max_length=max_title_length,
        truncation=True,
        add_special_tokens=False
    )[:max_title_length]
    title_padded = title_ids + [tokenizer.pad_token_id] * (max_title_length - len(title_ids))

    # 编码正文（保持content可生成max_content_length tokens）
    content_ids = tokenizer.encode(
        sample['content'],
        max_length=max_content_length,
        truncation=True,
        add_special_tokens=False
    )[:max_content_length]

    # 输入构造：标题 + 正文（前n-1 tokens）
    input_ids = title_padded + content_ids[:-1]

    # 输出构造：忽略标题部分（用-100） + 正文后n tokens作预测目标
    labels = [-100] * max_title_length + content_ids[1:]

    # 填充到总长度（title+content部分的总长度固定）
    total_length = max_title_length + max_content_length
    input_ids += [tokenizer.pad_token_id] * (total_length - len(input_ids))
    labels += [-100] * (total_length - len(labels))

    return input_ids, labels


# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset(batch_size, tokenizer, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(batch_size):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


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


# 文本生成测试代码
def generate_sentence(openings, model, tokenizer, window_size):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过120字则终止迭代
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


def train(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    char_dim = 768  # 每个字的维度
    window_size = 10  # 样本文本长度
    vocab_size = 21128  # 字表大小
    learning_rate = 0.001  # 学习率
    # 建立字表
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    # 加载语料
    corpus = load_corpus(corpus_path)

    # 建立模型
    model = LanguageModel(char_dim, vocab_size)
    if torch.cuda.is_available():
        model = model.cuda()
    # 建立优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, tokenizer, window_size, corpus)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("随着抽签仪式的临近，", model, tokenizer, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("json", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == '__main__':
    train("data/sample_data.json", False)
