'''
Author: Zhao
Date: 2025-02-11 15:42:06
LastEditTime: 2025-02-12 13:58:45
FilePath: homework.py
Description: 
'''
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np
import math
import random
import os
import re
import logging

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LanguageModel(nn.Module):
    def __init__(self, vocab):
        super(LanguageModel, self).__init__()
        bert_path = r"E:\bert-base-chinese"

        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.encoder = BertModel.from_pretrained(bert_path, return_dict=False)  
        self.hidden_size = self.encoder.config.hidden_size # 获取 BERT 的 hidden_size
        
        self.linear = nn.Linear(self.hidden_size, self.tokenizer.vocab_size)  # 输出BERT词表大小

        self.dropout = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # BERT 模型的输出
        input_ids = x
        attention_mask = create_causal_mask(input_ids) # 生成mask
        # 判断是否有GPU 确保都在同一设备上运行
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        y_pred = self.linear(outputs[0])    # Shape: (batch_size, seq_len, vocab_size)

        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.size(-1)), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)

#建立模型
def build_model(vocab):
    model = LanguageModel(vocab)
    return model

def create_causal_mask(input_ids):
    batch_size, seq_length = input_ids.size()
    causal_mask = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.bool))
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, seq_len, seq_len)
    return causal_mask

# 读取文件
def build_vocab(file_path):
    vocab = {"<pad>": 0, "[CLS]": 1, "[SEP]": 2}
    with open(file_path, encoding='utf8') as f:
        for index, line in enumerate(f):
            char = line.strip()  # 去掉结尾换行符和前后空格
            vocab[char] = index + 3  # 留出0位给pad token，1位给[CLS]，2位给[SEP]
    return vocab

# 处理输入
def process_input(openings, vocab, window_size):
    # 添加 [CLS] 标记
    input_ids = [vocab["[CLS]"]] + [vocab.get(char, vocab["<pad>"]) for char in openings[-window_size:]] + [vocab["[SEP]"]]
    return torch.LongTensor([input_ids])

#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:  # 统一编码
        for line in f:
            corpus += line.strip()
    return corpus

# 随机生成一个样本
# 从语料中随机选择一个位置，取出窗口大小的词作为输入，取出下一个词作为输出
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1: end + 1]
    #logger.info("Sample window: %s, %s", window, target)
    x = [vocab.get(word,vocab["<UNK>"]) for word in window]
    y = [vocab.get(word,vocab["<UNK>"]) for word in target]
    return x, y


# 生成一个batch的数据
# 生成batch_size个样本
# 每个样本的窗口大小为window_size
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 生成句子
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((index, char) for char, index in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成一个句子，直到生成换行符或者生成的字符数超过30
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = process_input(openings, vocab, window_size)
            if torch.cuda.is_available():
                x = x.cuda()
            # 预测下一个字符
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = reverse_vocab[index]
    return openings

# 采样策略
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

# 训练模型
def train(save_weight=True):
    global reverse_vocab 
    epoch_num = 100  # 训练轮数
    batch_size = 64 # batch大小
    window_size = 10 # 窗口大小
    #char_dim = 256  # 字符维度
    train_sample = 50000 # 训练样本数
    vocab = build_vocab("week10/data/vocab.txt")
    reverse_vocab = {idx: char for char, idx in vocab.items()} # 反转词表
    corpus = load_corpus("week10/data/corpus.txt")
    model = build_model(vocab)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4) # 优化器
    logger.info("本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus) # 生成一个batch的数据
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        avg_loss = np.mean(watch_loss)
        logger.info(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        logger.info(f"Generated Sentence: {generate_sentence('让他在半年之前，就不能做出', model, vocab, window_size)}")
        logger.info(f"Generated Sentence: {generate_sentence('李慕站在山路上，深深的呼吸', model, vocab, window_size)}")
    if not save_weight:
        return
    else:
        torch.save(model.state_dict(), "week10/model/nnlm.pth")
        logger.info("模型已保存")
        return
   
if __name__ == "__main__":
    train(save_weight=False)
