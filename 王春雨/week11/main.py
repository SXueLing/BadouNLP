#coding:utf8
import json
import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel
from loader import load_data
import logging
from config import Config

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
基于pytorch的LSTM语言模型
"""
pretrain_model_path =  r"D:\八斗\课件\第六周 语言模型\bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)

        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)

        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None,causal_mask=None):
        if y is not None:
            #训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])),29)
            #print(type(mask))
            #print(x[0])
            #print('======')
            # 使用列表推导式生成 attention_mask
            # 使用 torch.where 根据条件创建新的张量
            attention_mask = torch.where(x ==  tokenizer.get_vocab()["[SEP]"] , torch.tensor(0), x)
            expanded = attention_mask.unsqueeze(1)  # 添加第二维度
            temp_mask = (expanded == 0)
            mask_expanded = temp_mask.expand_as(mask)
            mask[mask_expanded] = 0 
            #print(attention_mask[0])
            #attention_mask = torch.where(x > 1, torch.tensor(1), attention_mask)

            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            #预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)



#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  #输入输出错开一位

    x = tokenizer.encode(window, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)   #将字转换成序号
    y = tokenizer.encode(target, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)

    return x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(batch_size, tokenizer, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(batch_size):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab_size, hidden_size, pretrain_model_path):
    model = LanguageModel(hidden_size, vocab_size, pretrain_model_path)
    return model

def create_causal_mask(seq_len):
    """创建下三角因果掩码"""
    mask = torch.tril(torch.ones(seq_len, seq_len)) == 1
    return mask.float()

def generate(model, tokenizer, title, max_length=64):
    model.eval()
    prompt = f"[CLS]{title}[SEP]"
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    
    generated = input_ids
    for _ in range(max_length):
        # 创建因果掩码
        seq_len = generated.shape[1]
        causal_mask = create_causal_mask(seq_len).unsqueeze(0).to(device)
        
        # 前向计算
        with torch.no_grad():
            logits = model(generated, causal_mask=causal_mask)
        
        # 取最后一个token预测
        next_token = torch.argmax(logits[0, -1, :])
        generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        # 遇到[SEP]停止
        if next_token == tokenizer.sep_token_id:
            break
    
    # 解码输出
    full_text = tokenizer.decode(generated[0], skip_special_tokens=False)
    content = full_text.split("[SEP]")[1].replace("[CLS]", "").strip()
    return content



#文本生成测试代码
def generate_sentence(openings, model, tokenizer, window_size):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 50:
            openings += pred_char
            x = tokenizer.encode(openings, add_special_tokens=False)
            #x.append(tokenizer.vocab["[SEP]"])
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
    epoch_num = 90        #训练轮数
    batch_size = 128       #每次训练样本个数
    train_sample = 10000   #每轮训练总共训练的样本总数
    hidden_size = 768        #每个字的维度
    window_size = 300       #样本文本长度
    vocab_size = 21128      #字表大小
    learning_rate = 0.001  #学习率


    train_data = load_data(r"sample_data.json", Config, logger,tokenizer)

    print(type(train_data))
    #corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(vocab_size, hidden_size, pretrain_model_path)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for index, batch_data in enumerate(train_data):
            if torch.cuda.is_available():
               batch_data = [d.cuda() for d in batch_data]
            x = batch_data[0]
            optim.zero_grad()    #梯度归零
            loss = model(x, x)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("呼唤和培育新型官德", model, tokenizer, window_size))
  
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
   
    train(r"corpus.txt", False)
    
