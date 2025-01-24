# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于 PyTorch 的 RNN 多分类网络
判断 "你" 在字符串的第几个位置
"""


class RNNModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # Embedding 层
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)  # RNN 层
        self.fc = nn.Linear(vector_dim, sentence_length + 1)  # 全连接层
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # 当输入真实标签时，返回 loss 值；无真实标签时，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        _, h_n = self.rnn(x)  # (batch_size, sen_len, vector_dim) -> (batch_size, vector_dim)
        x = self.fc(h_n.squeeze(0))  # (batch_size, vector_dim) -> (batch_size, sen_len + 1)
        if y is not None:
            return self.loss(x, y)  # 计算 loss
        else:
            return torch.argmax(x, dim=1)  # 返回预测类别


def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]  # 随机生成字符串
    if "你" in x:
        y = x.index("你")  # "你" 出现的位置
    else:
        y = sentence_length  # 未出现 "你"
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_model(vocab, char_dim, sentence_length):
    model = RNNModel(char_dim, sentence_length, vocab)
    return model


def evaluate(model, vocab, sample_length, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)
    print("测试集中类别分布：", torch.bincount(y))
    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        correct = (y_pred == y).sum().item()
    accuracy = correct / len(y)
    print(f"准确率：{accuracy:.4f}")
    return accuracy


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率

    vocab = build_vocab()  # 字表
    model = build_model(vocab, char_dim, sentence_length)  # 模型
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 优化器

    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)  # 计算 loss
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均 loss: {np.mean(watch_loss):.4f}")
        acc = evaluate(model, vocab, sentence_length, sentence_length)
        log.append([acc, np.mean(watch_loss)])

    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()


def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    x = []
    for input_string in input_strings:
        # 将输入字符串转换为索引，并进行填充或截断
        indices = [vocab.get(char, vocab['unk']) for char in input_string]
        if len(indices) < sentence_length:
            indices += [vocab["pad"]] * (sentence_length - len(indices))  # 填充
        else:
            indices = indices[:sentence_length]  # 截断
        x.append(indices)

    model.eval()  # 测试模式
    with torch.no_grad():
        results = model(torch.LongTensor(x))  # 模型预测

    for i, input_string in enumerate(input_strings):
        predicted_class = results[i].item()
        print(f"输入：{input_string}, 预测位置：{predicted_class if predicted_class < sentence_length else '未出现'}")


if __name__ == "__main__":
    main()
    test_strings = ["fn你vfe", "wz你dfg", "rqwdeg", "n我k你"]
    predict("model.pth", "vocab.json", test_strings)
