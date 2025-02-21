#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel
import json



class BERTClassifier(nn.Module):
    def __init__(self, pretrain_model_path, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # 使用BERT的[CLS]输出
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss
        return logits

# class LanguageModel(nn.Module):
#     def __init__(self, hidden_size, vocab_size, pretrain_model_path):
#         super(LanguageModel, self).__init__()
#         # self.embedding = nn.Embedding(len(vocab), input_dim)
#         # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)

#         self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')

#         self.classify = nn.Linear(hidden_size, vocab_size)
#         self.loss = nn.functional.cross_entropy

#     #当输入真实标签，返回loss值；无真实标签，返回预测值
#     def forward(self, x, y=None):
#         if y is not None:
#             #训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
#             mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
#             if torch.cuda.is_available():
#                 mask = mask.cuda()
#             x, _ = self.bert(x, attention_mask=mask)
#             y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
#             return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
#         else:
#             #预测时，可以不使用mask
#             x, _ = self.bert(x)
#             y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
#             return torch.softmax(y_pred, dim=-1)

#加载字表
# def build_vocab(vocab_path):
#     vocab = {"<pad>":0}
#     with open(vocab_path, encoding="utf8") as f:
#         for index, line in enumerate(f):
#             char = line[:-1]       #去掉结尾换行符
#             vocab[char] = index + 1 #留出0位给pad token
#     return vocab

#划分数据集
def train_test_split(data, labels, test_size=0.2, random_state=None):
    
    if len(data) != len(labels):
        raise ValueError("数据集和标签的长度必须相同！")
    
    # 设置随机种子
    if random_state is not None:
        random.seed(random_state)
    
    # 创建索引列表并打乱顺序
    indices = list(range(len(data)))
    random.shuffle(indices)
    
    # 计算划分点
    split_point = int(len(data) * (1 - test_size))
    
    # 划分训练集和测试集
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    
    # 根据索引提取数据和标签
    train_data = [data[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    return train_data, test_data, train_labels, test_labels

#数据预处理
def preprocess_data(tokenizer, data, max_length=512):
    """
    将文本和标签转换为BERT输入格式
    """
    texts = [item["title"] + " " + item["content"] for item in data]
    labels = [item["label"] for item in data]  # 假设数据中已经包含标签字段
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    labels = torch.tensor(labels)
    return inputs, labels

#加载语料
def load_json_file(file_path):
    
    data = []
    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

    # corpus = ""
    # with open(path, encoding="gbk") as f:
    #     for line in f:
    #         corpus += line.strip()
    # return corpus

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
def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# #建立模型
# def build_model(vocab, char_dim, pretrain_model_path):
#     model = LanguageModel(char_dim, vocab, pretrain_model_path)
#     return model

#文本生成测试代码
def generate_sentence(openings, model, tokenizer, window_size):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = tokenizer.encode(openings, add_special_tokens=False)
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



def train(json_path, save_weight=True):
    epoch_num = 10
    batch_size = 32
    learning_rate = 2e-5
    pretrain_model_path = "D:/AIClass/第六周 语言模型/bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    num_classes = 2  # 假设是二分类任务

    # 加载标注数据
    data = load_json_file(json_path)
    inputs, labels = preprocess_data(tokenizer, data)

    train_inputs, val_inputs, train_labels, val_labels = train_test_split(inputs, labels, test_size=0.2, random_state=42)


    # 构建模型
    model = BERTClassifier(pretrain_model_path, num_classes)
    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     inputs = {k: v.cuda() for k, v in inputs.items()}
    #     labels = labels.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("开始训练")
    for epoch in range(epoch_num):
        model.train()
        loss_list = []
        for i in range(0, len(train_labels), batch_size):
            batch_inputs = {k: v[i:i+batch_size] for k, v in train_inputs.items()}
            batch_labels = train_labels[i:i+batch_size]
            optimizer.zero_grad()
            loss = model(**batch_inputs, labels=batch_labels)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {np.mean(loss_list)}")

        model.eval()
        with torch.no_grad():
            val_loss = model(**val_inputs, labels=val_labels)
            print(f"Epoch {epoch+1}, Validation Loss: {val_loss.item()}")

    if save_weight:
        model_path = "model/classifier.pth"
        torch.save(model.state_dict(), model_path)
        print(f"模型权重已保存到 {model_path}")

    # epoch_num = 20        #训练轮数
    # batch_size = 128       #每次训练样本个数
    # train_sample = 10000   #每轮训练总共训练的样本总数
    # char_dim = 768        #每个字的维度
    # window_size = 10       #样本文本长度
    # vocab_size = 21128      #字表大小
    # learning_rate = 0.001  #学习率
    
    # pretrain_model_path = r'D:/AIClass/第六周 语言模型/bert-base-chinese'
    # tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    # json_file = load_json_file(json_path)     #加载语料
    # model = build_model(vocab_size, char_dim, pretrain_model_path)    #建立模型
    # # if torch.cuda.is_available():
    # #     model = model.cuda()
    # optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    # print("文本词表模型加载完毕，开始训练")
    # for epoch in range(epoch_num):
    #     model.train()
    #     watch_loss = []
    #     for batch in range(int(train_sample / batch_size)):
    #         x, y = build_dataset(batch_size, tokenizer, window_size, json_file) #构建一组训练样本
    #         if torch.cuda.is_available():
    #             x, y = x.cuda(), y.cuda()
    #         optim.zero_grad()    #梯度归零
    #         loss = model(x, y)   #计算loss
    #         loss.backward()      #计算梯度
    #         optim.step()         #更新权重
    #         watch_loss.append(loss.item())
    #     print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    #     print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer, window_size))
    #     print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, window_size))
    # if not save_weight:
    #     return
    # else:
    #     base_name = os.path.basename(corpus_path).replace("txt", "pth")
    #     model_path = os.path.join("model", base_name)
    #     torch.save(model.state_dict(), model_path)
    #     return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train(r"D:/AIClass/第十一周 大模型相关内容第一讲/homework/sample_data.json", True)
