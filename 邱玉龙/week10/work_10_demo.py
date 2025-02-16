#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel, BertConfig

"""
基于pytorch的BERT语言模型
"""

class BertLanguageModel(nn.Module):
    def __init__(self, bert_name=r'D:\learning\week6\xiawu\bert-base-chinese'):
        super(BertLanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_name,return_dict=False)
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.classify = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.loss = nn.functional.cross_entropy

    def forward(self, input_ids, attention_mask, labels=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        logits = self.classify(bert_output)
        
        if labels is not None:
            # 只计算被mask位置的loss
            loss = self.loss(logits.view(-1, logits.shape[-1]), labels.view(-1))
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

#修改build_sample函数，使用mask方式构建训练样本
def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - window_size)
    text = corpus[start:start + window_size]
    
    # 编码文本
    encoded = tokenizer(text, return_tensors='pt', padding='max_length', 
                       max_length=window_size, truncation=True)
    
    input_ids = encoded['input_ids'][0]
    attention_mask = encoded['attention_mask'][0]
    labels = input_ids.clone()
    
    # 创建三角形mask矩阵
    seq_length = input_ids.size(0)
    mask_matrix = torch.zeros(seq_length)
    
    # 设置每个位置的mask概率
    for i in range(seq_length):
        if i < seq_length // 4:
            mask_matrix[i] = 0.1  # 第一行概率较小
        elif i < seq_length // 2:
            mask_matrix[i] = 0.5  # 中间行概率适中
        else:
            mask_matrix[i] = 0.9  # 最后几行概率较大
    
    # 根据概率进行mask
    masked_indices = torch.bernoulli(mask_matrix).bool()
    input_ids[masked_indices] = tokenizer.mask_token_id
    
    # 将不需要预测的位置的label设为-100（PyTorch中交叉熵损失会忽略-100的位置）
    labels[~masked_indices] = -100
    
    return input_ids, attention_mask, labels

#修改build_dataset函数
def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_input_ids = []
    dataset_attention_mask = []
    dataset_labels = []
    
    for i in range(sample_length):
        input_ids, attention_mask, labels = build_sample(tokenizer, window_size, corpus)
        dataset_input_ids.append(input_ids)
        dataset_attention_mask.append(attention_mask)
        dataset_labels.append(labels)
    
    return (torch.stack(dataset_input_ids), 
            torch.stack(dataset_attention_mask), 
            torch.stack(dataset_labels))

#修改generate_sentence函数
def generate_sentence(opening, model, max_length=30):
    model.eval()
    with torch.no_grad():
        input_ids = model.tokenizer(opening, return_tensors='pt')['input_ids']
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        
        for _ in range(max_length):
            attention_mask = torch.ones_like(input_ids)
            outputs = model(input_ids, attention_mask=attention_mask)
            next_token_logits = outputs[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            
            if next_token == model.tokenizer.sep_token_id:
                break
                
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(input_ids.device)], dim=1)
        
        return model.tokenizer.decode(input_ids[0], skip_special_tokens=True)

def train(corpus_path, save_weight=True):
    epoch_num = 10
    batch_size = 30
    train_sample = 10000
    window_size = 50
    model = BertLanguageModel()  # 初始化模型
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 使用更小的学习率
    optim = torch.optim.AdamW(model.parameters(), lr=2e-4)
    
    corpus = load_corpus(corpus_path)
    print("BERT模型加载完毕，开始训练")
    
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            input_ids, attention_mask, labels = build_dataset(
                batch_size, model.tokenizer, window_size, corpus)
            
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()
            
            optim.zero_grad()
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model))
    
    if save_weight:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("corpus.txt", False)
