
import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, BertTokenizer

"""
基于pytorch的BERT语言模型,自回归
"""


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, input_dim):
        super(LanguageModel, self).__init__()
        # 使用BERT模型进行初始化
        self.bert = BertModel.from_pretrained(r'E:\BaiduNetdiskDownload\八斗精品课nlp\第六周 语言模型\bert-base-chinese')
        self.bert.resize_token_embeddings(vocab_size)  # 调整词表大小
        self.classify = nn.Linear(input_dim, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y=None, attention_mask=None):
        # 获取BERT的输出 (last_hidden_state)
        # attention_mask =
        output = self.bert(x, attention_mask=attention_mask) # shape: (batch_size, seq_len, vocab_size)
        logits = self.classify(output.last_hidden_state)  # shape: (batch_size, seq_len, vocab_size)

        if y is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            return loss
        else:
            return torch.softmax(logits, dim=-1)


# 加载字表
def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  # 去掉结尾换行符
            vocab[char] = index  #
    return vocab


# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位
    x = [vocab.get(word, 100) for word in window]  # 将字转换成序号
    y = [vocab.get(word, 100) for word in target]
    # x = tokenizer.encode(window, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)   #将字转换成序号
    # y = tokenizer.encode(target, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)
    return x, y


# 建立数据集
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def create_attention_mask(batch_size, seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))  # 下三角
    # mask = torch.triu(torch.ones(seq_len, seq_len))  # 上三角
    # mask = mask.masked_fill(mask == 1, float('-inf'))  # 将上三角部分填充为-inf
    # mask = mask.masked_fill(mask == 0,   1 )  # 将上三角部分填充为-inf

    return mask.unsqueeze(0).expand(batch_size, -1, -1)  # 生成与batch_size一致的mask



# 文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = [vocab.get(char, 100) for char in openings[-window_size:]]
            x = torch.LongTensor([x])


            if torch.cuda.is_available():
                x = x.cuda()

            y = model(x, attention_mask=None)[0][-1]
            # print(y)
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


# 计算文本ppl
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
            attention_mask = create_attention_mask(x.size(1)).unsqueeze(0)
            if torch.cuda.is_available():
                x = x.cuda()
                attention_mask = attention_mask.cuda()
            pred_prob_distribute = model(x, attention_mask=attention_mask)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 20        # 训练轮数
    batch_size = 64       # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    char_dim = 768        # 每个字的维度
    window_size = 10      # 样本文本长度
    # vocab = build_vocab("vocab.txt")  # 建立字表
    vocab = build_vocab(r"E:\BaiduNetdiskDownload\八斗精品课nlp\第六周 语言模型\bert-base-chinese/vocab.txt")  # 建立字表

    corpus = load_corpus(corpus_path)  # 加载语料
    model = LanguageModel(len(vocab), char_dim)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus)  # 构建一组训练样本
            attention_mask = create_attention_mask(batch_size, window_size)  # 为当前批次生成mask
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
                attention_mask = attention_mask.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, y, attention_mask=attention_mask)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss)}")
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
    if save_weight:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    train("corpus.txt", False)

