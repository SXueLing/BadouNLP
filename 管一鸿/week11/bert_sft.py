# coding:utf8

import torch.nn as nn
import numpy as np
import os
import re
from transformers import BertTokenizer, BertModel

"""
基于pytorch的LSTM语言模型
"""

import json
import torch
from typing import List, Dict
import random


def process_json_data(json_file, tokenizer, max_length=512):
    processed_data = []

    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # 解析JSON
                item = json.loads(line.strip())
                content = item['content']
                title = item['title']

                # 拼接文本
                full_text = content + '[SEP]' + title + '[EOS]'
                # 使用tokenizer处理
                encoded = tokenizer(
                    full_text,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

                # 获取input_ids和attention_mask
                input_ids = encoded['input_ids'].squeeze()
                attention_mask = encoded['attention_mask'].squeeze()

                # 创建labels，复制input_ids
                labels = input_ids.clone()

                # 找到第一个[SEP]的位置
                sep_pos = (input_ids == tokenizer.sep_token_id).nonzero()

                if len(sep_pos) > 0:
                    sep_pos = sep_pos[0][0]  # 获取第一个[SEP]的位置
                else:
                    sep_pos = input_ids.shape[0]  # 如果没有找到[SEP]，假设[SEP]后面没有有效的标签

                # 将[SEP]之前的所有标签设置为-100（包括[CLS]）
                labels[:sep_pos + 1] = -100

                # 将数据添加到列表
                processed_data.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                })

            except json.JSONDecodeError:
                print(f"跳过无效的JSON行: {line.strip()}")
            except Exception as e:
                print(f"处理数据时出错: {str(e)}")

    return processed_data


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        seq_len = x.shape[1]
        if y is not None:
            # 训练阶段
            # 生成SFT attention mask，控制哪些部分要关注
            attention_mask = self.create_sft_attention_mask(x,seq_len)

            if torch.cuda.is_available():
                attention_mask = attention_mask.cuda()

            # BERT前向传播
            x, _ = self.bert(x, attention_mask=attention_mask)
            y_pred = self.classify(x)  # 输出形状: (batch_size, seq_len, vocab_size)

            # 计算损失
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))  # 损失计算

        else:
            # 预测阶段
            # 使用SFT attention mask（通常可以去掉mask，因为生成时不需要自回归依赖）
            attention_mask = self.create_sft_attention_mask(x,seq_len)

            # BERT前向传播
            x, _ = self.bert(x, attention_mask=attention_mask)
            y_pred = self.classify(x)  # 输出形状: (batch_size, seq_len, vocab_size)

            # 返回预测的softmax分布
            return torch.softmax(y_pred, dim=-1)

    def create_sft_attention_mask(self, input_ids, seq_len):
        batch_size = input_ids.shape[0]
        sft_mask = torch.ones(batch_size, seq_len, seq_len).float()

        # 遍历每个样本
        for i in range(batch_size):
            sep_pos = (input_ids[i] == 4).nonzero()  # 4是[SEP]的token id
            if len(sep_pos) > 0:
                sep_pos = sep_pos[0][0]  # 获取[SEP]的位置
            else:
                sep_pos = seq_len  # 如果没有[SEP]，默认为整个序列长度

            # 对正文部分进行因果遮掩：创建下三角矩阵
            sft_mask[i, :sep_pos, :sep_pos] = torch.tril(torch.ones(sep_pos, sep_pos))

            # 对标题部分（[SEP]后）进行完全遮掩：不允许它们与正文部分交互
            sft_mask[i, sep_pos + 1:, :] = 0
            sft_mask[i, :, sep_pos + 1:] = 0

        return sft_mask  # 返回自定义的attention mask


# 加载字表
# def build_vocab(vocab_path):
#     vocab = {"<pad>":0}
#     with open(vocab_path, encoding="utf8") as f:
#         for index, line in enumerate(f):
#             char = line[:-1]       #去掉结尾换行符
#             vocab[char] = index + 1 #留出0位给pad token
#     return vocab

# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位

    x = tokenizer.encode(window, add_special_tokens=False, padding='max_length', truncation=True,
                         max_length=10)  # 将字转换成序号
    y = tokenizer.encode(target, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)

    return x, y


# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, pretrain_model_path):
    model = LanguageModel(768, 21128, pretrain_model_path)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, tokenizer, max_length=512):
    model.eval()
    with torch.no_grad():
        # 初始化生成文本
        generated_text = openings  # 使用给定的开头文本
        input_ids = tokenizer.encode(openings, add_special_tokens=False, truncation=True, max_length=max_length)

        # 将输入文本转换为张量
        input_ids = torch.LongTensor([input_ids])  # shape: (1, seq_len)

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        # 生成标题部分
        for _ in range(max_length):  # 生成最多 max_length 个token
            # 获取模型的输出
            outputs = model(input_ids)
            logits = outputs[0]  # (batch_size, seq_len, vocab_size)
            pred_token_logits = logits[0, -1, :]  # 获取最新预测token的logits，shape: (vocab_size,)

            # 使用sampling或greedy策略选取下一个token
            index = sampling_strategy(pred_token_logits)
            generated_text += tokenizer.decode([index])

            # 更新input_ids，为下一个token做准备
            input_ids = torch.cat([input_ids, torch.LongTensor([[index]])], dim=1)

            # 如果生成的文本已经超过了30个字符或生成了换行符，则停止
            if generated_text[-1] == "\n" or len(generated_text) > 30:
                break

    return generated_text


def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"  # Greedy策略
    else:
        strategy = "sampling"  # 随机采样策略

    if strategy == "greedy":
        # greedy：选择最大概率的token
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        # sampling：从概率分布中按概率采样
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


def train(data_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 128  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    char_dim = 768  # 每个字的维度
    vocab_size = 21128  # 字表大小
    learning_rate = 0.001  # 学习率

    # 预训练模型路径
    pretrain_model_path = r'D:\Badou\ner\pretrained_models\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    # 加载处理好的数据
    data = process_json_data(data_path, tokenizer)

    # 构建模型
    model = build_model(vocab_size, char_dim, pretrain_model_path)
    if torch.cuda.is_available():
        model = model.cuda()

    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("文本词表模型加载完毕，开始训练")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        for batch in range(int(train_sample / batch_size)):
            
            # 这里直接从处理后的数据中获取batch数据
            batch_data = data[batch * batch_size:(batch + 1) * batch_size]
            if len(batch_data):
                # 从数据中获取input_ids, attention_mask, labels
                x = torch.stack([torch.tensor(item['input_ids']) for item in batch_data])
                y = torch.stack([torch.tensor(item['labels']) for item in batch_data])
                attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch_data])

                # 将数据移动到GPU（如果可用）
                if torch.cuda.is_available():
                    x, y, attention_mask = x.cuda(), y.cuda(), attention_mask.cuda()

                optim.zero_grad()  # 梯度归零

                # 计算损失
                loss = model(x, y)  # 计算loss

                loss.backward()  # 计算梯度
                optim.step()  # 更新权重

                watch_loss.append(loss.item())

        print(f"=========\n第{epoch + 1}轮平均loss: {np.mean(watch_loss)}")

    # 如果需要保存模型权重
    if not save_weight:
        return
    else:
        base_name = os.path.basename(data_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train(r"./sample_data.json", False)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # data = process_json_data('sample_data.json', tokenizer)
    # print(data[0])
