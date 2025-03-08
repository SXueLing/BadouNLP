# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
from transformers import BertModel, BertTokenizer

"""
基于 BERT 的 SFT（Supervised Fine-Tuning）实现
任务：文本分类（判断文本是否包含某些特定字符）
"""

class TorchModel(nn.Module):
    def __init__(self, num_classes):
        super(TorchModel, self).__init__()
        # 加载预训练 BERT 模型
        self.bert = BertModel.from_pretrained("bert-base-chinese", return_dict=False)
        # 添加分类层
        self.classify = nn.Linear(self.bert.config.hidden_size, num_classes)
        # 使用交叉熵损失
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, y=None):
        # BERT 前向传播
        sequence_output, pooler_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 分类层
        logits = self.classify(pooler_output)
        if y is not None:
            # 计算损失
            return self.loss(logits, y.squeeze())
        else:
            # 返回预测结果
            return logits


# 使用 BERT 的 Tokenizer 处理输入
def build_sample(sentence_length):
    # 随机生成一个句子
    chars = random.choices("abcdefghijklmnopqrstuvwxyz", k=sentence_length)
    text = "".join(chars)
    # 使用 BERT Tokenizer 编码
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    inputs = tokenizer(text, padding='max_length', max_length=sentence_length, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"].squeeze(0)  # 去掉 batch 维度
    attention_mask = inputs["attention_mask"].squeeze(0)  # 去掉 batch 维度
    # 标签逻辑保持不变
    if set("abc") & set(chars) and not set("xyz") & set(chars):
        y = 0  # A 类样本
    elif not set("abc") & set(chars) and set("xyz") & set(chars):
        y = 1  # B 类样本
    else:
        y = 2  # C 类样本
    return input_ids, attention_mask, y


# 建立数据集
def build_dataset(sample_length, sentence_length):
    dataset_input_ids = []
    dataset_attention_mask = []
    dataset_y = []
    for _ in range(sample_length):
        input_ids, attention_mask, y = build_sample(sentence_length)
        dataset_input_ids.append(input_ids)
        dataset_attention_mask.append(attention_mask)
        dataset_y.append(y)
    return (
        torch.stack(dataset_input_ids),  # 输入 ID
        torch.stack(dataset_attention_mask),  # 注意力掩码
        torch.LongTensor(dataset_y)  # 标签
    )


# 测试代码
def evaluate(model, sentence_length):
    model.eval()
    total = 200  # 测试样本数量
    input_ids, attention_mask, y = build_dataset(total, sentence_length)
    correct, wrong = 0, 0
    with torch.no_grad():
        logits = model(input_ids, attention_mask)  # 模型预测
        y_pred = torch.argmax(logits, dim=1)  # 获取预测类别
        correct = (y_pred == y).sum().item()
        wrong = total - correct
    print("正确预测个数：%d / %d, 正确率：%f" % (correct, total, correct / total))
    return correct / total


def main():
    epoch_num = 15  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    sentence_length = 6  # 样本文本长度
    num_classes = 3  # 分类类别数

    # 建立模型
    model = TorchModel(num_classes)
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)  # 优化器

    # 训练过程
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            # 构建训练数据
            input_ids, attention_mask, y = build_dataset(batch_size, sentence_length)
            optim.zero_grad()  # 梯度归零
            loss = model(input_ids, attention_mask, y)  # 计算损失
            loss.backward()  # 反向传播
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    return


if __name__ == "__main__":
    main()
