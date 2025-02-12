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
基于pytorch的LSTM语言模型
"""




class LanguageModel(nn.Module):
    def __init__(self, vocab, input_dim):
        super(LanguageModel, self).__init__()
        
        # BERT Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')  # 使用中文预训练BERT
        
        # BERT模型
        self.bert = BertModel.from_pretrained('bert-base-chinese')  # 加载中文BERT模型
        
        # 定义全连接层，用于输出预测
        self.classify = nn.Linear(input_dim, len(vocab))
        
        self.loss = nn.CrossEntropyLoss()  # 计算loss

    def forward(self, x, attention_mask=None, labels=None):
        # 使用BERT模型进行编码
        output = self.bert(x, attention_mask=attention_mask)
        
        # 获取每个token的输出 (batch_size, seq_len, hidden_size)
        sequence_output = output.last_hidden_state
        # 通过全连接层得到预测值
        logits = self.classify(sequence_output)

        # 如果有真实标签（labels），计算损失
        if labels is not None:
            # 展平 logits 和 labels
            logits = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
            labels = labels.view(-1)  # (batch_size * seq_len,)

            loss = self.loss(logits, labels)
            return loss
        else:
            return torch.softmax(logits, dim=-1)

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
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(vocab, window_size, corpus, tokenizer):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位
    x = tokenizer(window, padding='max_length', truncation=True, max_length=window_size, return_tensors="pt")
    y = [vocab.get(word, vocab["<UNK>"]) for word in target]
    return x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, vocab, window_size, corpus, tokenizer):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus, tokenizer)
        dataset_x.append(x['input_ids'].squeeze(0))  # 取出input_ids
        dataset_y.append(y)
    return torch.stack(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim):
    model = LanguageModel( vocab,char_dim)
    return model

#文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
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




#计算文本ppl
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
    return 2 ** (prob * ( -1 / len(sentence)))


def train(corpus_path, save_weight=False):
    epoch_num = 20
    batch_size = 64
    train_sample = 50000
    char_dim = 768
    window_size = 10
    vocab = build_vocab("vocab.txt")
    corpus = load_corpus(corpus_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = build_model(vocab, char_dim)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus, tokenizer)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            attention_mask = (x != tokenizer.pad_token_id).long()  # 生成attention_mask
            optim.zero_grad()
            loss = model(x, attention_mask=attention_mask, labels=y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
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
