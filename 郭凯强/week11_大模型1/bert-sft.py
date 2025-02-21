#coding:utf8    #指定文件编码为UTF-8

import json    #导入JSON处理库
import torch    #导入PyTorch深度学习框架
import torch.nn as nn    #导入神经网络模块
import numpy as np    #导入数值计算库
import math    #导入数学运算库
import random    #导入随机数生成库
import os    #导入操作系统接口库
import re    #导入正则表达式库
from transformers import BertTokenizer, BertModel    #导入BERT模型相关组件
from torch.utils.data import Dataset, DataLoader    #导入数据加载器

"""
基于Bert结构，进行sft形式的训练
SFT (Supervised Fine-Tuning) 是指在预训练模型基础上进行有监督微调
"""

class LanguageModel(nn.Module):    #定义语言模型类
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):    #初始化函数
        super(LanguageModel, self).__init__()    #调用父类初始化
        # 注释掉原始的Embedding和LSTM层，改用BERT
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)

        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')
        
        self.classify = nn.Linear(hidden_size, vocab_size)    #输出层，将BERT的输出映射到词表大小
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)    #损失函数，忽略标签为-1的位置

    def forward(self, x, mask=None, y=None):    #前向传播函数
        if y is not None:    #训练模式
            #训练时，使用mask矩阵控制注意力范围
            print(mask.shape)    #打印mask的形状，用于调试
            x, _ = self.bert(x, attention_mask=mask)    #通过BERT处理输入
            y_pred = self.classify(x)    #生成预测结果
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))    #计算损失
        else:    #预测模式
            x, _ = self.bert(x)    #通过BERT处理输入
            y_pred = self.classify(x)    #生成预测结果
            return torch.softmax(y_pred, dim=-1)    #返回概率分布

def load_corpus(path):    #加载训练语料
    corpus = []    #初始化语料列表
    with open(path, encoding="utf8") as f:    #打开语料文件
        for line in f:    #逐行读取
            line = json.loads(line)    #解析JSON格式
            corpus.append([line["title"], line["content"]])    #添加标题和内容对
    return corpus    #返回语料列表

def build_dataset(tokenizer, corpus, max_length, batch_size):    #构建数据集
    dataset = []    #初始化数据集列表
    for i, (prompt, answer) in enumerate(corpus):    #遍历语料
        # 对输入文本进行编码
        prompt_encode = tokenizer.encode(prompt, add_special_tokens=False)    #编码prompt
        answer_encode = tokenizer.encode(answer, add_special_tokens=False)    #编码answer
        
        # 构建输入序列：[CLS] prompt [SEP] answer [SEP]
        x = [tokenizer.cls_token_id] + prompt_encode + [tokenizer.sep_token_id] + answer_encode + [tokenizer.sep_token_id]
        
        # 构建标签序列：prompt部分用-1标记（不参与损失计算），answer部分保持原值
        y = len(prompt_encode) * [-1] + [-1] + answer_encode + [tokenizer.sep_token_id] + [-1]
        
        # 构建注意力掩码
        mask = create_mask(len(prompt_encode), len(answer_encode))    #创建掩码矩阵
        
        # 对序列进行padding
        x = x[:max_length] + [0] * (max_length - len(x))    #补齐输入序列
        y = y[:max_length] + [0] * (max_length - len(y))    #补齐标签序列
        
        # 转换为张量
        x = torch.LongTensor(x)    #转换输入为张量
        y = torch.LongTensor(y)    #转换标签为张量
        mask = pad_mask(mask, (max_length, max_length))    #补齐掩码矩阵
        
        dataset.append([x, mask, y])    #添加到数据集
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)    #返回数据加载器

def create_mask(s1, s2):    #创建注意力掩码
    len_s1 = s1 + 2    #prompt长度（包含CLS和SEP）
    len_s2 = s2 + 1    #answer长度（包含SEP）
    mask = torch.ones(len_s1 + len_s2, len_s1 + len_s2)    #创建全1矩阵
    
    # 设置掩码规则
    for i in range(len_s1):    #遍历prompt部分
        mask[i, len_s1:] = 0    #prompt不能看到answer部分
    
    for i in range(len_s2):    #遍历answer部分
        mask[len_s1 + i, len_s1 + i + 1:] = 0    #answer只能看到当前及之前的token
    
    return mask    #返回掩码矩阵

def pad_mask(tensor, target_shape):    #补齐掩码矩阵
    height, width = tensor.shape    #获取原始形状
    target_height, target_width = target_shape    #目标形状
    result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)    #创建目标大小的零矩阵
    
    # 计算有效区域
    h_end = min(height, target_height)    #高度取最小值
    w_end = min(width, target_width)    #宽度取最小值
    
    # 复制有效区域
    result[:h_end, :w_end] = tensor[:h_end, :w_end]    #复制数据
    return result    #返回补齐后的矩阵

def build_model(vocab, char_dim, pretrain_model_path):    #构建模型
    model = LanguageModel(768, 21128, pretrain_model_path)    #创建模型实例
    return model    #返回模型

def generate_sentence(openings, model, tokenizer):    #生成文本
    model.eval()    #设置为评估模式
    openings = tokenizer.encode(openings)    #编码输入文本
    
    with torch.no_grad():    #不计算梯度
        while len(openings) <= 50:    #生成不超过50个token
            x = torch.LongTensor([openings])    #构建输入张量
            if torch.cuda.is_available():    #如果有GPU
                x = x.cuda()    #将数据移到GPU
            y = model(x)[0][-1]    #获取预测结果
            index = sampling_strategy(y)    #采样下一个token
            openings.append(index)    #添加到序列中
    
    return tokenizer.decode(openings)    #解码生成的文本

def sampling_strategy(prob_distribution):    #采样策略
    if random.random() > 0.1:    #90%概率使用贪婪搜索
        strategy = "greedy"
    else:    #10%概率使用随机采样
        strategy = "sampling"
    
    if strategy == "greedy":    #贪婪搜索
        return int(torch.argmax(prob_distribution))    #返回概率最大的token
    elif strategy == "sampling":    #随机采样
        prob_distribution = prob_distribution.cpu().numpy()    #转换为numpy数组
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)    #按概率采样

def main(corpus_path, save_weight=True):    #主函数
    # 设置训练参数
    epoch_num = 20        #训练轮数
    batch_size = 32       #批次大小
    char_dim = 768        #字符维度
    max_length = 50       #最大序列长度
    vocab_size = 21128    #词表大小
    learning_rate = 0.001 #学习率
    
    # 加载预训练模型和分词器
    pretrain_model_path = r'/Users/cagiant/Downloads/第六周 语言模型/bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    
    # 准备训练数据和模型
    corpus = load_corpus(corpus_path)    #加载语料
    train_data = build_dataset(tokenizer, corpus, max_length, batch_size)    #构建数据集
    model = build_model(vocab_size, char_dim, pretrain_model_path)    #构建模型
    
    if torch.cuda.is_available():    #如果有GPU
        model = model.cuda()    #将模型移到GPU
    
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)    #创建优化器
    
    print("文本词表模型加载完毕，开始训练")    #打印训练开始信息
    
    # 开始训练循环
    for epoch in range(epoch_num):    #遍历每个epoch
        model.train()    #设置为训练模式
        watch_loss = []    #记录损失值
        
        for x, mask, y in train_data:    #遍历训练数据
            if torch.cuda.is_available():    #如果有GPU
                x, mask, y = x.cuda(), mask.cuda(), y.cuda()    #数据移到GPU
            
            optim.zero_grad()    #清空梯度
            loss = model(x, mask, y)    #计算损失
            loss.backward()    #反向传播
            optim.step()    #更新参数
            watch_loss.append(loss.item())    #记录损失值
        
        # 打印训练信息和生成样例
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("北京明年拟推工作日半价观看电影", model, tokenizer))
        print(generate_sentence("南京一合金厂锅炉发生爆炸", model, tokenizer))
    
    # 保存模型
    if not save_weight:    #如果不保存权重
        return
    else:    #如果需要保存权重
        base_name = os.path.basename(corpus_path).replace("txt", "pth")    #构建保存文件名
        model_path = os.path.join("model", base_name)    #构建保存路径
        torch.save(model.state_dict(), model_path)    #保存模型参数
        return

if __name__ == "__main__":    #程序入口
    main("sample_data.json", False)    #运行主函数