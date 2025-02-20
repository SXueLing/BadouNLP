# coding:utf8
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertForMaskedLM

from config import Config

"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), input_dim)
        self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        self.classify = nn.Linear(input_dim, len(vocab))
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # output shape:(batch_size, sen_len, input_dim)
        x, _ = self.layer(x)  # output shape:(batch_size, sen_len, input_dim)
        y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


class AutoregressiveDataset(torch.utils.data.Dataset):
    def __init__(self, titles, contents, tokenizer, max_length=512):
        self.titles = titles
        self.contents = contents
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        title = self.titles[idx]
        content = self.contents[idx]

        # 拼接标题和内容
        input_text = title + " [SEP] " + content
        encoding = self.tokenizer(input_text, truncation=True, padding='max_length', max_length=self.max_length,
                                  return_tensors='pt', add_special_tokens=False)
        input_ids = encoding['input_ids'].squeeze(0)  # [B, T] => [T]

        # 设置标签：标题部分的标签是ignore_index（通常为-100），内容部分的标签是移位后的输入
        labels = input_ids.clone()

        # 标题部分的标签设置为ignore_index
        title_end_idx = len(title)  # 获取标题的长度
        labels[:title_end_idx] = -100  # 将标题部分的标签设置为-100

        # 内容部分的标签设置为移位后的输入
        labels[title_end_idx:] = input_ids[title_end_idx - 1:-1]

        # 创建因果mask，标题后开始的地方应用掩码
        attention_mask = torch.tril(torch.ones(self.max_length, self.max_length))  # Lower triangular matrix
        return input_ids, labels, attention_mask


class BertModelA(nn.Module):
    def __init__(self, config, l_v):
        super(BertModelA, self).__init__()
        model_dir = config["bert_path"]
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir=model_dir)
        self.bert = BertModel.from_pretrained("bert-base-chinese", cache_dir=model_dir)
        # self.bert.config.num_layers = config["num_layers"]
        # self.bert.config.hidden_size = config["hidden_size"]
        # self.bert.class_num = config["class_num"]
        self.classify = nn.Linear(self.bert.config.hidden_size, l_v)
        # self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss()  # loss采用交叉熵损失

    def forward(self, x, target=None):
        x = self.bert(input_ids=x).last_hidden_state  # Get the last hidden state
        y_pred = self.classify(x)
        # print(predict.shape)
        if target is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), target.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


# 加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>": 0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  # 去掉结尾换行符
            vocab[char] = index + 1  # 留出0位给pad token
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
# def build_sample(vocab, window_size, corpus):
#     start = random.randint(0, len(corpus) - 1 - window_size)
#     end = start + window_size
#     window = corpus[start:end]
#     target = corpus[start + 1:end + 1]  # 输入输出错开一位
#     # print(window, target)
#     x = [vocab.get(word, vocab["<UNK>"]) for word in window]  # 将字转换成序号
#     y = [vocab.get(word, vocab["<UNK>"]) for word in target]
#     return x, y

def build_sample(window_size, corpus, tokenizer):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位

    # print(window)
    # print(target)
    x = tokenizer.encode(window, add_special_tokens=False)
    y = tokenizer.encode(target, add_special_tokens=False)

    # print(window, target)
    # x = [vocab.get(word, vocab["<UNK>"]) for word in window]  # 将字转换成序号
    # y = [vocab.get(word, vocab["<UNK>"]) for word in target]
    return x, y


# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset(sample_length, window_size, corpus):
    dataset_x = []
    dataset_y = []
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", cache_dir=Config["bert_path"])
    for i in range(sample_length):
        x, y = build_sample(window_size, corpus, tokenizer)
        if (len(x) == len(y) and len(x) == window_size):
            dataset_x.append(x)
            dataset_y.append(y)

    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, l_v):
    # model = LanguageModel(char_dim, vocab)
    model = BertModelA(Config, l_v)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", cache_dir=Config["bert_path"])

    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 60:
            openings += pred_char
            # x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
            x = tokenizer.encode(openings, add_special_tokens=False)
            x = torch.LongTensor([x])
            print(x.shape)
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)
            logits = y.logits[:,-1,:].squeeze()

            index = sampling_strategy(logits[-1])
            pred_char = tokenizer.convert_ids_to_tokens(index)
            # pred_char = reverse_vocab[index]
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
        return int(np.random.choice(list(range(len(prob_distribution))), p=prob_distribution))


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
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))


def load_data():
    d = json.load(open(r"D:\pythonProject\NLP\week10\transformers-生成文章标题\sample_data.json", encoding="utf-8"))
    title = []
    content = []
    for i in d:
        title.append(i["title"])
        content.append(i["content"])

    return title, content


def train(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 1  # 每次训练样本个数
    train_sample = 100  # 每轮训练总共训练的样本总数
    char_dim = 256  # 每个字的维度
    window_size = 512  # 样本文本长度
    vocab = build_vocab("vocab.txt")  # 建立字表
    l_v = 21128
    # corpus = load_corpus(corpus_path)  # 加载语料
    # model = build_model(vocab, char_dim, l_v)  # 建立模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    titles, content = load_data()
    model = BertForMaskedLM.from_pretrained("bert-base-chinese", cache_dir=Config["bert_path"])
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", cache_dir=Config["bert_path"])
    dataset = AutoregressiveDataset(titles, content, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):  # 训练3轮
        model.train()
        for input_ids, labels, attention_mask in dataloader:
            input_ids, labels, attention_mask = input_ids.to(device), labels.to(device), attention_mask.to(device)

            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss

            # 反向传播
            optim.zero_grad()
            loss.backward()
            optim.step()

            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        # print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        # print(generate_sentence("北京明年拟推工作日半价观看电影[SEP]", model, vocab, window_size))
        input_ids = tokenizer.encode("北京明年拟推工作日半价观看电影[SEP]", return_tensors="pt").to(device)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)
        title_end_idx = input_ids[0].tolist().index(tokenizer.encode("[SEP]")[1])
        attention_mask[0, :title_end_idx] = 0
        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(generated_text)
        # print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("corpus.txt", False)
