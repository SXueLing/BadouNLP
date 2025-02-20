import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel
import json
# 内容长300， title 长20
class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)

        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        title = torch.LongTensor(x[:, -20:])

        mask = torch.ones((x.shape[0], x.shape[1], x.shape[1]))
        for i in range(x.shape[0]):
            for j in range(0, 320):
                for k in range(300, 320):
                    if (k > j):
                        mask[i, j, k] = 0

        if torch.cuda.is_available():
            mask = mask.cuda()
        x, _ = self.bert(x, attention_mask=mask)
        y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
        y_pred = y_pred[:, -20:, :]

        if y is not None:
            return self.loss(y_pred.reshape(-1, y_pred.shape[-1]), title.reshape(-1))
        else:
            return torch.softmax(y_pred, dim=-1)

def build_model( pretrain_model_path):
    model = LanguageModel(768, 21128, pretrain_model_path)
    return model

#加载语料
def load_data(path):
    data = []
    with open(path, encoding="utf8") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            title = line["title"]
            content = line["content"]
            data.append({ "title": title, "content": content })
    return data

def build_sample(tokenizer, content_size, title_size, data_item):
    contentToken = tokenizer.encode(data_item["content"], add_special_tokens=False, padding='max_length', truncation=True, max_length=content_size)
    titleToken = tokenizer.encode(data_item["title"], add_special_tokens=False, padding='max_length', truncation=True, max_length=title_size)
    return contentToken + titleToken

def build_dataset(sample_length, tokenizer, corpus_path):

    arr = load_data(corpus_path)
    # print(len(arr))
    temp = random.sample(arr, sample_length)
    # print([build_sample(tokenizer, 300, 20, x) for x in temp], 'aaa')
    return torch.LongTensor([build_sample(tokenizer, 300, 20, x) for x in temp])

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

def evaluate(model, tokenizer, corpus_path):
    arr = load_data(corpus_path)
    model.eval()
    item = random.choice(arr)
    predTitle = ''
    with torch.no_grad():
        print("content: ", item['content'])
        print("title: ", item['title'])
        y = model(torch.LongTensor([build_sample(tokenizer, 300, 20, item)]))
        for i in range(y.shape[1]):
            t = y[0][i]
            index = sampling_strategy(t)
            pred_char = ''.join(tokenizer.decode(index))
            predTitle = predTitle + pred_char
        print("predTitle: ", predTitle)
def train(corpus_path, save_weight=True):
    epoch_num = 10        #训练轮数
    batch_size = 6       #每次训练样本个数
    train_sample = 30   #每轮训练总共训练的样本总数
    char_dim = 768        #每个字的维度
    window_size = 10       #样本文本长度
    vocab_size = 21128      #字表大小
    learning_rate = 0.001  #学习率
    

    pretrain_model_path = r'D:\py\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    model = build_model( pretrain_model_path)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x = build_dataset(batch_size, tokenizer, corpus_path) #构建一组训练样本
            # print(x, 'xxxxxxxxxxxxxx')
            if torch.cuda.is_available():
                x = x.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, True)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        evaluate(model, tokenizer, corpus_path)
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return

if __name__ == "__main__":
    train("sample_data.json", False)