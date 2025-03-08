#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import matplotlib.pyplot as plt

from transformers import BertModel, BertTokenizer

"""
基于pytorch的语言模型，用bert+因果掩码实现，训练时间太长，看不到loss收敛后的结果
"""

bert_path = r"D:\work\data_receive\0109\bert-base-chinses"

class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        self.layer = BertModel.from_pretrained(r"D:\work\data_receive\0109\bert-base-chinses", return_dict=False)
        self.classify = nn.Linear(input_dim, len(vocab))
        self.classify.weight.data.normal_(mean=0.0, std=0.02)
        self.classify.bias.data.zero_()
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        batch_size, seq_len = x.shape
        # 生成因果掩码（下三角矩阵）
        causal_mask = torch.tril(torch.ones(batch_size, seq_len, seq_len)).bool()
        # 调用BERT并传入自定义掩码
        last_hidden_state, _ = self.layer(x, attention_mask=causal_mask)
        y_pred = self.classify(last_hidden_state)
        res = []
        for batch in y_pred:
            logits = batch[-2, :]  # 取最后一个时间步的输出,避开[SEP]
            predicted_index = sampling_strategy(logits)
            res.append(predicted_index)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1), ignore_index=0)
        else:
            return torch.softmax(y_pred, dim=-1)

#加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>":0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index + 1 #留出0位给pad token
    return vocab

#加载语料
def load_corpus(path):
    corpus = []
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus.append(line.strip())
    return corpus

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(corpus, window=10):
    tokenizer = BertTokenizer.from_pretrained(r"D:\work\data_receive\0109\bert-base-chinses")
    text = random.choice(corpus)
    if len(text) <= window:
        return None, None
    input_text = text[:window]
    output_text = text[1:window+1]
    x = tokenizer.encode(input_text, max_length=window+2, padding='max_length', truncation=True)
    y = tokenizer.encode(output_text, max_length=window+2, padding='max_length', truncation=True)
    return x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(corpus)
        if x is not None and y is not None:
            dataset_x.append(x)
            dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model

#文本生成测试代码
def generate_sentence(openings, model, vocab, window_size, max_length=30):
    tokenizer = BertTokenizer.from_pretrained(r"D:\work\data_receive\0109\bert-base-chinses")
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    pred_char = ""

    with torch.no_grad():
        while pred_char != "\n" and len(openings) <= max_length:
            openings += pred_char
            input_tokens = openings[-window_size:] if len(openings) > window_size else openings
            x = tokenizer.encode(input_tokens, max_length=30, pad_to_max_length=True, truncation=True)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0]  # 取最后一个时间步的输出
            logits = y[-2, :]  # 取最后一个时间步的输出
            predicted_index = sampling_strategy(logits)
            pred_char = tokenizer.decode([predicted_index])
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



#计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["[UNK]"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["[UNK]"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 2)
    return 2 ** (prob * ( -1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 100        #训练轮数
    batch_size = 32       #每次训练样本个数
    train_sample = 2000   #每轮训练总共训练的样本总数
    char_dim = 768        #每个字的维度
    window_size = 10       #样本文本长度
    vocab = build_vocab(r"D:\work\data_receive\0109\bert-base-chinses\vocab.txt")       #建立字表
    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(vocab, char_dim)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    for param in model.layer.parameters():
        param.requires_grad = False
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    flag = False
    for epoch in range(epoch_num):
        print("=====================================")
        print(epoch)
        print("=====================================")
        model.train()
        if epoch >= 3:
            flag = True
        for i in range(6,9):
            for param in model.layer.encoder.layer[i].parameters():
                param.requires_grad = flag
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, corpus) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        print(generate_sentence("你好", model, vocab, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        # model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), base_name)
        return

def calculate_text_lengths(texts):
    lengths = [len(text) for text in texts]
    return lengths


def analyze_text_lengths(texts):
    lengths = calculate_text_lengths(texts)
    average_length = np.mean(lengths)
    print(f"Average text length: {average_length}")

    # 90% percentile
    truncate_length_90 = np.percentile(lengths, 90)
    print(f"90% percentile truncate length: {truncate_length_90}")

    # 75% percentile
    truncate_length_75 = np.percentile(lengths, 75)
    print(f"75% percentile truncate length: {truncate_length_75}")

    # Plot histogram
    plt.hist(lengths, bins=20)
    plt.title("Text Length Distribution")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("corpus.txt", True)
    # corpus = load_corpus("corpus.txt")
    # analyze_text_lengths(corpus)
