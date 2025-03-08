# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
from transformers import BertTokenizer, BertModel

"""
基于 BERT 的自回归语言模型
"""

class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        # 加载预训练 BERT 模型
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)
        # 分类层，将 BERT 的输出映射到词汇表大小
        self.classify = nn.Linear(hidden_size, vocab_size)
        # 损失函数
        self.loss = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充 token 的损失

    def forward(self, x, y=None):
        if y is not None:
            # 训练时，构建下三角掩码矩阵，模拟单向注意力
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)  # 输出形状：(batch_size, seq_len, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # 预测时，不使用掩码
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # 输出形状：(batch_size, seq_len, vocab_size)
            return torch.softmax(y_pred, dim=-1)

# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

# 随机生成一个样本
def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位
    x = tokenizer.encode(window, add_special_tokens=False)[:window_size]
    y = tokenizer.encode(target, add_special_tokens=False)[:window_size]
    return x, y

# 建立数据集
def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    # 动态填充到最大长度
    max_len = max(len(seq) for seq in dataset_x)
    dataset_x = [seq + [0] * (max_len - len(seq)) for seq in dataset_x]
    dataset_y = [seq + [0] * (max_len - len(seq)) for seq in dataset_y]
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 文本生成测试代码
def generate_sentence(openings, model, tokenizer, max_len=30):
    model.eval()
    with torch.no_grad():
        pred_text = openings
        while len(pred_text) < max_len:
            inputs = tokenizer.encode(pred_text, add_special_tokens=False)
            inputs = torch.LongTensor([inputs])
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            outputs = model(inputs)
            next_token_logits = outputs[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            next_token = tokenizer.decode([next_token_id])
            if next_token == "[SEP]" or next_token == "[PAD]":
                break
            pred_text += next_token
    return pred_text

# 训练过程
def train(corpus_path, save_weight=True):
    epoch_num = 10          # 训练轮数
    batch_size = 32         # 每次训练样本个数
    train_sample = 5000     # 每轮训练总共训练的样本总数
    window_size = 20        # 样本文本长度
    vocab_size = 21128      # 字表大小
    learning_rate = 5e-5    # 学习率

    pretrain_model_path = r'E:\BaiduNetdiskDownload\人工智能课程\第六周 语言模型\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    corpus = load_corpus(corpus_path)     # 加载语料
    model = LanguageModel(768, vocab_size, pretrain_model_path)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # 建立优化器

    print("模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, tokenizer, window_size, corpus)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算 loss
            loss.backward()  # 反向传播
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均 loss: {np.mean(watch_loss)}")
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer))

    if save_weight:
        base_name = os.path.basename(corpus_path).replace(".txt", ".pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    train("corpus.txt", False)
