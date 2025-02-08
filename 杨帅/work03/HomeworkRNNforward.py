# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

"""
手动实现简单的神经网络
使用pytorch实现RNN
手动实现RNN
对比
"""


class TorchRNN(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        """
        RNN 模型
        :param vector_dim:输入数据大小
        :param sentence_length:文本长度
        :param vocab:将文本准备转换为的数据 list
        """
        super(TorchRNN, self).__init__()

        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.pool = nn.AvgPool1d(sentence_length)
        self.layer = nn.RNN(vector_dim, vector_dim, bias=False, batch_first=True)
        self.layer_class = nn.Linear(vector_dim, sentence_length + 1)
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    def forward(self, x, y=None):
        x = self.embedding(x)  # embedding  训练批次的数量 * 文本最大字符长度 -> 训练批次的数量 * 文本最大字符长度 * 指定单字的向量长度
        out, h = self.layer(x)  # RNN 网络训练 训练批次的数量 * 文本最大字符长度 * 指定单字的向量长度 -> 指定单字的向量长度 * 1
        x = out[:, -1, :]
        # out = self.layer(x)  # RNN 网络训练 训练批次的数量  * 指定单字的向量长度 -> 指定单字的向量长度 * 1
        y_pred = self.layer_class(x)  # 归一化 指定单字的向量长度 * 1 -> 指定单字的向量长度 * 1

        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


# 字符集随便挑了一些字，实际上还可以扩充
# 为每个字生成一个标号
# {"a":1, "b":2, "c":3...}
# abc -> [1,2,3]
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 26
    return vocab


# 随机生成一个样本
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = []
    # 指定哪些字出现时为正样本
    y = np.zeros(sentence_length + 1)
    target_def = "你"
    key_list = list(vocab.keys())

    for i in range(sentence_length):
        ra_str = random.choice(key_list)
        x.append(ra_str)
        if ra_str == target_def:
            key_list.remove(target_def)

    if target_def in x:
        indices = np.where(np.array(x) == target_def)[0]
        y[indices[0]] = 1

    count_of_ones = np.sum(y)
    if count_of_ones == 0:
        y[sentence_length] = 1

    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y


# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchRNN(char_dim, sentence_length, vocab)
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            p_var, p_index = torch.max(y_p, dim=0)

            indices_of_ones = np.where(y_t == 1)[0]
            if p_index in indices_of_ones:
                correct += 1  # 判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def train_main():
    # 训练 入口方法
    # 配置参数
    epoch_num = 50  # 训练轮数
    batch_size = 200  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 开始训练
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for i in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算损失
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    print("")
    pass


def test_main():
    # 测试入口
    pass


if __name__ == '__main__':
    train_main()
    test_main()
