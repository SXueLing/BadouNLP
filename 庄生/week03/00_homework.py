#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务：判断文本中是否有某些特定字符出现

做一个多分类任务，判断特定字符k在字符串的第几个位置，使用rnn和交叉熵。
"""


def printSplit():
    print("-"*120)


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):  # 25  6  字符表
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  # embedding层
        # self.pool = nn.AvgPool1d(sentence_length)   # 池化层
        # print("!"*80,vector_dim) # 25
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)
        self.classify = nn.Linear(vector_dim, sentence_length+1)  # 线性层
        # self.activation = torch.sigmoid     # sigmoid归一化函数
        # self.loss = nn.functional.mse_loss  # loss函数采用均方差损失
        # self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # embedding 层
        # print("embedding前的x", x, x.shape)
        
        # embedding 是数字转向量
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        # print("embedding后的x", x, x.shape)  # 20 * 6 * 25

        # 先转，再平均池化
        # x = x.transpose(1, 2)  # (batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        # print("transpose后的x", x)
        # 池化
        # x = self.pool(x)  # (batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        # print("pool后的x", x)
        # 去一个维度
        # x = x.squeeze()  # (batch_size, vector_dim, 1) -> (batch_size, vector_dim)
        # print("squeeze后的x", x)
        # 以下经过神经网络处理了：
        output, h = self.rnn(x)
        x = output[:, -1, :]
        # printSplit()
        # print("output为：", output, output.shape)  # [20, 6, 25]
        # print("x为：", x, x.shape)  # [20, 25]
        # output = output[:, -1, :]
        y_pred = self.classify(x)  # (batch_size, vector_dim) -> (batch_size, 1) 3*20 20*1 -> 3*1
        # print("linear后的x", y_pred, y_pred.shape,y, y.shape)
        # y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        # print("activation后的x", x)
        if y is not None:

            # print("y为：", y, y.shape)  # [20, 7]
            # print("y_pred为：", y_pred, y_pred.shape)  # [20, 7]

            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:

            return y_pred  # 输出预测结果


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


# 字符集随便挑了一些字，实际上还可以扩充
# 为每个字生成一个标号
# {"a":1, "b":2, "c":3...}
# abc -> [1,2,3]
def build_vocab():
    """
        该函数用于生成字符对应的数字的字典
    :return: 键名为字符，值为对应数字的字典
    """
    chars = "你我他defghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 26
    return vocab


# 随机生成一个样本
# 从所有字中选取sentence_length个字
# 反之为负样本
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if random.choice([True, False]):
        x[random.choice([0, 1, 2, 3, 4])] = 'k'
    # 指定哪些字出现时为正样本
    try:
        y = x.index("k")
    except ValueError:
        y = sentence_length
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y


# 建立数据集-构建样本
# 输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    """
    :param sample_length:  这个数据集多少个样本（多少行文字/多少句话）
    :param vocab: 字符对应表
    :param sentence_length:  一行文字/一句话有多少个字符
    :return: 
    """
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)

    # print(dataset_y)
    # print(torch.LongTensor(y_true))
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# vocab = build_vocab()
# build_dataset(20, vocab, 6)
# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)  # 建立200个用于测试的样本
    # print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), 200 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比

            max_value, max_index = torch.max(y_p, dim=0)
            if max_index == y_t:
                correct += 1  # 样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    # 配置参数
    epoch_num = 20         # 训练轮数
    batch_size = 20        # 每次训练样本个数
    train_sample = 500     # 每轮训练总共训练的样本总数
    char_dim = 25          # 每个字的维度
    sentence_length = 6    # 样本文本长度
    learning_rate = 0.005  # 学习率
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
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            # print("!!!!!!!!!!!!!1", x, x.shape)
            optim.zero_grad()    # 梯度归零
            loss = model(x, y)   # 计算loss
            loss.backward()      # 计算梯度
            optim.step()         # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)     # 建立模型
    model.load_state_dict(torch.load(model_path))             # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()   # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    # for i, input_string in enumerate(input_strings):
        # print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i]))  # 打印结果


if __name__ == "__main__":
    a = 1;
    main()
    test_strings = ["fnvfee", "wz你dfg", "rqwdeg", "n我kwww"]
    predict("model.pth", "vocab.json", test_strings)
