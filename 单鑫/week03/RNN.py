import json
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as Data

"""
构建一个 用RNN实现的 判断某个字符的位置 的任务

5 分类任务 判断 a出现的位置 返回index +1 or -1
"""


class TorchModel(nn.Module):
    def __init__(self, sentence_length, hidden_size, vocab, input_dim, output_size):
        super(TorchModel, self).__init__()
        self.emb = nn.Embedding(len(vocab) + 1, input_dim)
        self.rnn = nn.RNN(input_dim, hidden_size, batch_first=True)

        self.pool = nn.MaxPool1d(sentence_length)
        self.leaner = nn.Linear(hidden_size, output_size)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        # x = 15 * 4
        x = self.emb(x)  # output = 15 * 4 * 10
        x, h = self.rnn(x)  # output = 15 * 4 * 20 h = 1*15*20
        x = self.pool(x.transpose(1, 2)).squeeze()  # output = 15 * 20 * (1,被去除)
        y_pred = self.leaner(x)  # output = 15 * 5
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

    # 创建字符集 只有6个 希望a出现的概率大点


def build_vocab():
    chars = "abcdef"
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    # vocab['unk'] = len(vocab) + 1
    return vocab


# 构建样本集
def build_dataset(vocab, data_size, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(data_size):
        x, y = build_simple(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 构建样本
def build_simple(vocab, sentence_length):
    # 随机生成 长度为4的字符串
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if x.count('a') != 0:
        y = x.index('a')
    else:
        y = 4

    # 转化为 数字
    x = [vocab[char] for char in list(x)]
    return x, y


def main():
    batch_size = 15
    simple_size = 500
    vocab = build_vocab()
    # 每个样本的长度为4
    sentence_length = 4
    # 样本的向量维度为10
    input_dim = 10
    # rnn的隐藏层 随便设置为20
    hidden_size = 20
    # 5 分类任务
    output_size = 5
    # 学习率
    lr = 0.02
    # 轮次
    epoch_size = 25
    model = TorchModel(sentence_length, hidden_size, vocab, input_dim, output_size)

    # 优化函数
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # 样本
    x, y = build_dataset(vocab, simple_size, sentence_length)
    dataset = Data.TensorDataset(x, y)
    dataiter = Data.DataLoader(dataset, batch_size)
    for epoch in range(epoch_size):
        epoch_loss = []
        model.train()
        for x, y_true in dataiter:
            loss = model(x, y_true)
            loss.backward()
            optim.step()
            optim.zero_grad()
            epoch_loss.append(loss.item())
        print("第%d轮 loss = %f" % (epoch + 1, np.mean(epoch_loss)))
        # evaluate
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果

    return


# 评估效果
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(vocab, 200, sentence_length)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d / %d, 正确率：%f" % (correct, correct + wrong, correct / (correct + wrong)))
    return correct / (correct + wrong)


if __name__ == '__main__':
    main()
