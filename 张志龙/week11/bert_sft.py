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
from torch.utils.data import Dataset, DataLoader

"""
基于Bert结构，进行sft形式的训练
"""

class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)

        # 加载预训练的bert模型
        # 参数说明：pretrain_model_path：预训练模型的路径，return_dict=False：返回的是tuple，attn_implementation='eager'：使用eager模式
        # The attention implementation to use in the model (if relevant). 
        # Can be any of `"eager"` (manual implementation of the attention), 
        # `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)),
        # or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). 
        # By default, if available, SDPA will be used for torch>=2.1.1.
        # The default is otherwise the manual `"eager"` implementation.
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')

        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)  # ignore_index=-1表示不计算-1的loss，sft仅对answer部分使用交叉熵损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, mask=None, y=None):
        if y is not None:
            # 训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
            # print("mask.shape:", mask.shape)   # mask.shape:（batch_size, seq_len, seq_len）
            # print("x.shape1:", x.shape)  # x.shape1:（batch_size, seq_len）
            x, _ = self.bert(x, attention_mask = mask)
            # print("x.shape2:", x.shape)  # x.shape2:（batch_size, seq_len, hidden_size）
            y_pred = self.classify(x)   # output shape:(batch_size, seq_len, vocab_size)
            # print("y_pred.shape:", y_pred.shape)  # y_pred.shape:（batch_size, seq_len, vocab_size）
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # 预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)   # output shape:(batch_size, seq_len, vocab_size)
            return torch.softmax(y_pred, dim=-1)  # 返回概率分布，shape:(batch_size, seq_len, vocab_size)=(1, title.len, 21128)

# 加载语料, 用title当成假想的prompt，content当成假想的answer
def load_corpus(path):
    corpus = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            corpus.append([line["title"], line["content"]])  # append的结果类似于[['title1', 'content1'], ['title2', 'content2'], ...]
    return corpus

# sft的数据构造
# loss只计算答案部分，通过mask矩阵，让上下文之间没有交互
# label中使用-1，表示不参与训练（交叉熵损失函数中ignore_index=-1）
# 	   cls	x1	x2	x3	sep	y1	y2	sep
# cls	1	1	1	1	1	0	0	0
# x1	1	1	1	1	1	0	0	0
# x2	1	1	1	1	1	0	0	0
# x3	1	1	1	1	1	0	0	0
# sep	1	1	1	1	1	0	0	0
# y1	1	1	1	1	1	1	0	0
# y2	1	1	1	1	1	1	1	0
# sep	1	1	1	1	1	1	1	1
								
# 输入	cls	x1	x2	x3	sep	y1	y2	sep
# 输出	x1	x2	x3	sep	y1	y2	sep	
# label	-1	-1	-1	-1	y1	y2	sep	-1

def build_dataset(tokenizer, corpus, max_length, batch_size):
    dataset = []
    for i, (prompt, answer) in enumerate(corpus):
        prompt_encode = tokenizer.encode(prompt, add_special_tokens=False)  # add_special_tokens=False表示不添加[CLS]和[SEP]
        answer_encode = tokenizer.encode(answer, add_special_tokens=False)  # add_special_tokens=False表示不添加[CLS]和[SEP]
        # 自行构建输入和输出，加上[CLS]和[SEP]，并且在prompt和answer之间加上[SEP]
        x = [tokenizer.cls_token_id] + prompt_encode + [tokenizer.sep_token_id] + answer_encode + [tokenizer.sep_token_id]
        y = len(prompt_encode) * [-1] + [-1] + answer_encode + [tokenizer.sep_token_id] + [-1]
        # 构建一个的mask矩阵，让prompt内可以交互，answer中上下文之间没有交互
        mask = create_mask(len(prompt_encode), len(answer_encode))
        # padding
        x = x[:max_length] + [0] * (max_length - len(x))
        y = y[:max_length] + [0] * (max_length - len(y))
        x = torch.LongTensor(x)  # 将list转换为tensor
        y = torch.LongTensor(y)
        mask = pad_mask(mask, (max_length, max_length))
        dataset.append([x, mask, y])
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
# 构造掩码，输入两个字符串的长度
def create_mask(s1, s2):
    len_s1 = s1 + 2 # cls + sep
    len_s2 = s2 + 1 # sep
    # 创建掩码张量
    mask = torch.ones(len_s1 + len_s2, len_s1 + len_s2)
    # 遍历s1的每个token
    # Prompt区域可见性控制
    for i in range(len_s1):
        # s1的当前token不能看到s2的任何token
        mask[i, len_s1:] = 0  
    # 遍历s2的每个token
    # Answer区域可见性控制
    for i in range(len_s2):
        # s2的当前token不能看到后面的s2 token
        mask[len_s1 + i, len_s1 + i + 1:] = 0
    return mask

def pad_mask(tensor, target_shape):
    # 获取输入张量和目标形状的长宽
    height, width = tensor.shape
    target_height, target_width = target_shape
    # 创建一个全零张量,形状为目标形状
    result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    # 计算需要填充或截断的区域
    h_start = 0
    w_start = 0
    h_end = min(height, target_height)
    w_end = min(width, target_width)
    # 将原始张量对应的部分填充到全零张量中
    result[h_start:h_end, w_start:w_end] = tensor[:h_end - h_start, :w_end - w_start]
    return result

# 建立模型
def build_model(vocab_size, char_dim, pretrain_model_path):
    model = LanguageModel(char_dim, vocab_size, pretrain_model_path)
    return model

# 文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    model.eval()  # 模型进入测试模式
    openings = tokenizer.encode(openings) # 将起始文本编码为id
    with torch.no_grad():    # 不计算梯度
        # 生成文本超过50字则终止迭代
        while len(openings) <= 50:
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            # print("model(x).shape:", model(x).shape)  # model(x): torch.Size([1, title.len, 21128])
            # print("model(x)[0].shape:", model(x)[0].shape)  # model(x)[0]: torch.Size([title.len, 21128])
            # print("model(x)[0][-1].shape:", model(x)[0][-1].shape)  # model(x)[0][-1]: torch.Size([21128])
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            # 如果预测的下标是UNK，则重新预测
            while index == 100:
                index = sampling_strategy(y)
                
            openings.append(index)  # 将预测的下标加入到openings中
    return tokenizer.decode(openings)  # 将openings解码为文本返回

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"    # 90%的概率使用贪心策略
    else:
        strategy = "sampling"  # 10%的概率使用随机采样策略
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))  # 返回概率最大的下标
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()  # 将概率分布转换为numpy数组
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)  # 返回一个随机采样的下标

def main(corpus_path, save_weight=True):
    epoch_num = 20        # 训练轮数
    batch_size = 32       # 每次训练样本个数
    char_dim = 768        # 每个字的维度
    max_length = 50       # 样本文本长度
    vocab_size = 21128    # 字表大小
    learning_rate = 0.001 # 学习率
    

    pretrain_model_path = r'./badou/week6/bert-base-chinese'
    # 加载预训练的tokenizer
    # bert模型和tokenizer的关系是：tokenizer将文本转换为id，模型将id转换为概率分布，tokenizer将概率分布转换为文本
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)     # 加载训练语料
    train_data = build_dataset(tokenizer, corpus, max_length, batch_size)  # 建立数据集
    model = build_model(vocab_size, char_dim, pretrain_model_path)    # 建立模型
    if torch.cuda.is_available():  # 如果有GPU，则将模型放到GPU上
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   # 使用Adam优化器，传入模型参数和学习率
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()  # 模型进入训练模式
        watch_loss = []  # 用于记录每个batch的loss
        for x, mask, y in train_data: # 遍历数据集，每次取出一个batch_size的数据
            if torch.cuda.is_available():
                x, mask, y = x.cuda(), mask.cuda(), y.cuda()   # 将数据放到GPU上
            optim.zero_grad()    # 梯度归零
            loss = model(x, mask, y)   # 计算loss
            loss.backward()      # 计算梯度
            optim.step()         # 更新权重
            watch_loss.append(loss.item())  # 记录loss
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("北京明年拟推工作日半价观看电影", model, tokenizer))  # 进行文本生成测试
        print(generate_sentence("南京一合金厂锅炉发生爆炸", model, tokenizer))
    if not save_weight:
        return
    else:  # 保存模型权重
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)  # 保存路径
        torch.save(model.state_dict(), model_path)  # 保存模型权重
        return

if __name__ == "__main__":
    main("./badou/week11 大语言模型相关第一讲/作业/sample_data.json", False)
