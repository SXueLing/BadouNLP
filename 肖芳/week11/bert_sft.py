#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel, BertConfig
import fancy_utils as utils
import json
from createSFTMask import createSFTMaskForY

"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, pretrain_model_path):
        super(LanguageModel, self).__init__()


        bertConfig = BertConfig.from_pretrained(pretrain_model_path)
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')

        self.classify = nn.Linear(hidden_size, bertConfig.vocab_size)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        if y is not None:
            # 计算有几个-100（截断位置）
            # 训练时，构建一个符合sft的mask矩阵
            mask = createSFTMaskForY(y)
            if torch.backends.mps.is_available():
                mask = mask.to('mps')
                x = x.to('mps')
                y = y.to('mps')
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            # 计算loss的时候忽略0
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1), ignore_index=-100) 
        else:
            #预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)

#加载训练数据
def load_train_data(path):
    datas = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            datas.append(json.loads(line))
    return datas

#根据一行训练数据 生成一对X Y
def build_sample(tokenizer, data):
    # title, content
    title = data['title']
    content = data['content']
    
    #将字转换成序号

    x_text = f"[CLS]{title}[SEP]{content}[SEP]"
    y_text = f"{title}[SEP]{content}[SEP]"

    x = tokenizer.encode(x_text, add_special_tokens=False, padding='max_length', truncation=True, max_length=500)  
    y = tokenizer.encode(y_text, add_special_tokens=False, padding='max_length', truncation=True, max_length=500)
    len_title = len(title)
    y[:len_title] = [-100]*len_title

    return x, y

#建立数据集
def build_dataset(tokenizer, datas):
    dataset_x = []
    dataset_y = []
    for data in datas:
        x, y = build_sample(tokenizer, data)
        dataset_x.append(x)
        dataset_y.append(y)

    # print("dataset x:",dataset_x[0], "\n y:",dataset_y[0])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(char_dim, pretrain_model_path):
    model = LanguageModel(char_dim, pretrain_model_path)
    return model


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


#文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = tokenizer.encode(openings, add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.backends.mps.is_available():
                x = x.to('mps')
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = ''.join(tokenizer.decode(index))
    return openings

def train(train_data_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 52       #每次训练样本个数
    train_sample = 104   #每轮训练总共训练的样本总数
    char_dim = 768        #每个字的维度
    learning_rate = 0.00001  #学习率
    

    pretrain_model_path = utils.BERT_PATH
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    datas = load_train_data(train_data_path)     #加载训练数据
    x, y = build_dataset(tokenizer, datas)
    
    model = build_model(char_dim, pretrain_model_path)    #建立模型
    if torch.backends.mps.is_available():
        model = model.to('mps')
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    print("已加载训练数据")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        
        for batch in range(int(train_sample / batch_size)):
            start_index = batch * batch_size
            end_index = start_index + batch_size
            batch_x = x[start_index:end_index]
            batch_y = y[start_index:end_index]
            if torch.backends.mps.is_available():
                x, y = x.to('mps'), y.to('mps')
            optim.zero_grad()    #梯度归零
            loss = model(batch_x, batch_y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer))


    if not save_weight:
        return
    else:
        torch.save(model.state_dict(), 'model.pth')
        return

if __name__ == "__main__":
    train_path = "sample_data.json"
    train(train_path, save_weight=False)
