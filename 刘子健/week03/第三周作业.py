# coding: utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中特定字符（'你'、'我'、'他'）出现的位置

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab_size):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(vector_dim, 50, batch_first=True)  # 添加一个简单的RNN层
        self.classify = nn.Linear(50, sentence_length + 1)  # 输出层，多分类，类别数为句子长度+1（包含特殊类别）
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x, _ = self.rnn(x)  # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, hidden_dim)
        x = x.mean(dim=1)  # 对RNN的输出进行平均，得到每个句子的表示（或者可以选择其他池化方式）
        x = self.classify(x)  # (batch_size, hidden_dim) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(x, y)  # 计算损失
        else:
            return x  # 输出预测结果（logits）

# 字符集和标号生成
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 未知字符的序号
    return vocab

# 随机生成一个样本
def build_sample(vocab, sentence_length, target_chars):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 找到目标字符在字符串中的位置，如果没有则返回一个特殊值（这里选择sentence_length + 1）
    positions = [i for i, char in enumerate(x) if char in target_chars]
    if positions:
        y = random.choice(positions)  # 随机选择一个目标字符的位置作为标签（为了简化任务，只选择一个）
    else:
        y = sentence_length  # 特殊值，表示目标字符不在字符串中
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y

# 建立数据集
def build_dataset(sample_length, vocab, sentence_length, target_chars):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length, target_chars)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 建立模型
def build_model(vocab, char_dim, sentence_length):
    vocab_size = len(vocab)
    model = TorchModel(char_dim, sentence_length, vocab_size)
    return model

# 测试代码
def evaluate(model, vocab, sentence_length, target_chars, sample_length=200):
    model.eval()
    x, y_true = build_dataset(sample_length, vocab, sentence_length, target_chars)
    y_pred = model(x).argmax(dim=1)  # 获取预测的位置
    correct = (y_pred == y_true).sum().item()
    accuracy = correct / sample_length
    print(f"正确预测个数：{correct}, 正确率：{accuracy:.4f}")
    return accuracy

def main():
    # 配置参数
    epoch_num = 10        # 训练轮数
    batch_size = 20       # 每次训练样本个数
    train_sample = 500    # 每轮训练总共训练的样本总数
    char_dim = 20         # 每个字的维度
    sentence_length = 6   # 样本文本长度
    learning_rate = 0.005 # 学习率
    target_chars = {'你', '我', '他'}  # 目标字符集

    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length, target_chars)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss):.4f}")
        acc = evaluate(model, vocab, sentence_length, target_chars)
        log.append([acc, np.mean(watch_loss)])

    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    with open("vocab.json", "w", encoding="utf8") as writer:
        json.dump(vocab, writer, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

# 使用训练好的模型做预测（注意：这里的预测函数需要修改以适应多分类任务）
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    target_chars = {'你', '我', '他'}  # 目标字符集

    with open(vocab_path, "r", encoding="utf8") as f:
        vocab = json.load(f)  # 加载字符表

    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    predictions = []
    for input_string in input_strings:
        x = [vocab[char] for char in input_string] + [0] * (sentence_length - len(input_string))  # 填充到句子长度
        x = torch.LongTensor([x])  # 转换为张量，并增加batch维度
        with torch.no_grad():
            y_pred = model(x).argmax(dim=1)  # 获取预测的位置
        predictions.append((input_string, y_pred.item()))

    for input_string, pred_position in predictions:
        if pred_position < len(input_string):
            predicted_char = input_string[pred]
