# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch的网络编写
实现一个网络完成预测一个字符在一串字符串中的位置的NLP任务
"""


class PosPredictModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(PosPredictModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)  # 使用RNN层处理序列信息
        self.fc = nn.Linear(vector_dim, sentence_length)  # 输出层，预测位置，输出维度为字符串长度
        self.activation = nn.Softmax(dim=1)  # 使用Softmax将输出转换为概率分布
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    def forward(self, x, y=None):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # 取最后一个时间步的输出
        x = self.fc(x)
        x = self.activation(x)
        if y is not None:
            return self.loss(x, y)  # 计算预测结果与真实标签的损失
        return x


# 构建字符到索引的映射（词表）
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  # 字符集，可按需扩充
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


# 随机生成一个样本
def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    x = [vocab.get(word, vocab['unk']) for word in x]
    # 随机选取一个字符的位置作为标签（这里假设预测字符在字符串中的位置）
    pos = random.randint(0, sentence_length - 1)
    y = pos
    return x, y


# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = PosPredictModel(char_dim, sentence_length, vocab)
    return model


# 测试代码，用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    correct = 0
    total = 0
    with torch.no_grad():
        y_pred = model(x)
        _, predicted = torch.max(y_pred, 1)
        for pred, true in zip(predicted, y):
            total += 1
            if pred == true:
                correct += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / total))
    return correct / total


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 200  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    learning_rate = 0.5  # 学习率
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
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "pos_predict_model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])
    model.eval()
    with torch.no_grad():
        result = model(x)
    for i, input_string in enumerate(input_strings):
        _, predicted_pos = torch.max(result[i], 0)
        print("输入：%s, 预测位置：%d" % (input_string, predicted_pos.item()))


if __name__ == "__main__":
    main()
