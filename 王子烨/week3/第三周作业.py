# -*- coding: utf-8 -*-
# @Time    : 2024/12/4 15:31
# @Author  : yeye
# @File    : nlpdemo.py
# @Software: PyCharm
# @Desc    :
"""

"""
import json
import random

import numpy as np
import torch.optim
import torch.nn as nn

"""
基于pytorch框架
根据特定字符出现的位置判断是第几类
运用rnn和交叉熵
"""


class TorchModel(nn.Module):
    def __init__(self, vocab, vector_dim):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)
        self.classify = nn.Linear(vector_dim, 6)
        self.activation = torch.softmax
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)  # batch_size * sentence_length * vector_dim
        x, hidden_state = self.rnn(
            x)  # batch_size * sentence_length * vector_dim -> batch_size * sentence_length * hidden_size(vector_dim)
        hidden_state = hidden_state[-1, :, :]  # batch_size * vector_size
        y_pre = self.classify(hidden_state)
        y_pre = self.activation(y_pre,dim=-1)
        if y is not None:
            return self.loss(y_pre, y)
        else:
            y_pre, class_num = torch.max(y_pre, dim=1)
            return y_pre, class_num


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz你我他"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


def build_model(vocab, vector_dim):
    model = TorchModel(vocab, vector_dim)
    return model


def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length - 1)]
    x.append("你")
    random.shuffle(x)
    y = x.index("你")
    x = [vocab.get(char, vocab['unk']) for char in x]
    return x, y


def build_dataset(batch_size, vocab, sentence_length):
    X = []
    Y = []
    for i in range(batch_size):
        x, y = build_sample(vocab, sentence_length)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X), torch.LongTensor(Y)


def evalute(model, vocab):
    model.eval()
    x, y = build_dataset(100, vocab, 6)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pre, class_num = model(x)
        for y_p, y_t in zip(y, class_num):
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确个数：%d 正确概率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    epoch_num = 100  # 训练轮次
    char_dim = 20  # 字符维度
    sentence_length = 6  # 句子长度
    learning_rate = 1e-3
    train_sample = 500
    batch_size = 20
    vocab = build_vocab()
    model = build_model(vocab, char_dim)
    optim = torch.optim.Adam(model.parameters(), learning_rate)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for i in range(train_sample // batch_size):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evalute(model, vocab)
        log.append([acc, np.mean(watch_loss)])
    torch.save(model.state_dict(), 'model2.pth')
    writer = open('vocab.json', 'w', encoding='utf-8')
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


def predict(model_path, vocab_path, input_strings):
    vocab = json.load(open(vocab_path, 'r', encoding='utf-8'))
    char_dim = 20
    model = build_model(vocab, char_dim)
    model.load_state_dict(torch.load(model_path))
    X = []
    for input_string in input_strings:
        x = [vocab.get(char, vocab['unk']) for char in input_string]
        X.append(x)
    model.eval()
    with torch.no_grad():
        y_pre, class_nums = model(torch.LongTensor(X))
    for y_p, class_num, input_string in zip(y_pre, class_nums, input_strings):
        print("字符串：%s 预测概率值：%f 预测类别：%d" % (input_string, y_p, class_num))


if __name__ == '__main__':
    main()
    input_strings = ['asdsd你', '你asdsd', 'a你sdsd', 'asd你sd']
    predict('model2.pth', 'vocab.json', input_strings)
