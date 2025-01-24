# coding: utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch的网络编写
实现一个网络完成一个简单NLP任务
判断文本中特定字符（如'你'、'我'、'他'）出现的具体位置（多分类任务）
"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab_size, num_classes):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vector_dim, padding_idx=0)  # Embedding层
        self.rnn = nn.LSTM(vector_dim, 64, batch_first=True)  # 使用LSTM进行序列建模
        self.fc = nn.Linear(64, num_classes)  # 全连接层，输出类别数等于可能的位置数+1（不包含字符的类别）
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sentence_length) -> (batch_size, sentence_length, vector_dim)
        _, (hidden, _) = self.rnn(x)  # LSTM输出，我们只取最后一个时间步的隐藏状态
        hidden = hidden.squeeze(0)  # (batch_size, 1, 64) -> (batch_size, 64)
        logits = self.fc(hidden)  # (batch_size, 64) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(logits, y)  # 计算损失
        else:
            return logits  # 输出预测结果


# 构建字符到索引的映射
def build_vocab():
    chars = "你我他abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字符对应一个序号
    vocab['unk'] = len(vocab)  # 未知字符
    return vocab


# 构建训练样本
# 返回一个包含句子和标签（字符位置）的元组
def build_sample(vocab, sentence_length):
    sentence = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    positions = {char: idx for idx, char in enumerate(sentence) if char in {'你', '我', '他'}}
    if positions:
        label = min(positions.values())  # 取最小位置作为标签（简化问题，只预测第一个出现的位置）
    else:
        label = len(sentence)  # 如果字符都不存在，则标签为句子长度（作为特殊类别）
    x = [vocab.get(word, vocab['unk']) for word in sentence]  # 将字符转换为索引
    return x, label


# 构建数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 构建模型
def build_model(vocab, char_dim, sentence_length):
    vocab_size = len(vocab)
    num_classes = sentence_length + 1  # 类别数为句子长度+1（表示字符不在句子中）
    model = TorchModel(char_dim, sentence_length, vocab_size, num_classes)
    return model


# 评估模型
def evaluate(model, vocab, sentence_length, sample_length):
    model.eval()
    x, y = build_dataset(sample_length, vocab, sentence_length)
    correct = 0
    with torch.no_grad():
        preds = model(x)
        preds = preds.argmax(dim=1)  # 取预测概率最大的类别
        correct = (preds == y).sum().item()
    accuracy = correct / sample_length
    print(f"正确预测个数：{correct}, 正确率：{accuracy:.4f}")
    return accuracy


def main():
    # 配置参数
    epoch_num = 10
    batch_size = 20
    train_sample = 500
    char_dim = 20
    sentence_length = 6
    learning_rate = 0.005

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
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss):.4f}")
        acc = evaluate(model, vocab, sentence_length, batch_size * int(train_sample / batch_size))
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


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    with open(vocab_path, "r", encoding="utf8") as f:
        vocab = json.load(f)
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        for input_string in input_strings:
            x = [vocab[char] for char in input_string]
            x = torch.LongTensor([x + [0] * (sentence_length - len(x))])  # 填充到句子长度
            logits = model(x)
            pred = logits.argmax(dim=1).item()
            if pred < sentence_length:
                print(f"输入：{input_string}, 预测位置：{pred}")
            else:
                print(f"输入：{input_string}, 字符不在句子中")


if __name__ == "__main__":
    main()
    test_strings = ["fnvfee", "wz你dfg", "rqwdeg", "n我kwww", "abcdef"]
    predict("model.pth", "vocab.json", test_strings)
