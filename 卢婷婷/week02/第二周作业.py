# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的五分类任务

"""

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层，输出维度为类别数
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(logits, y)  # 计算交叉熵损失
        else:
            return logits

# 生成一个样本, 样本的生成方法代表了我们要学习的规律
# 这里我们随机生成一个5维向量，并随机分配一个0-4的类别标签
def build_sample():
    x = np.random.random(5)
    y = random.randint(0, 4)  # 随机生成0-4之间的整数作为标签
    return x, y

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 直接使用整数标签
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 标签转换为LongTensor

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct = 0
    with torch.no_grad():
        logits = model(x)  # 模型预测
        _, predicted = torch.max(logits, 1)  # 获取预测类别
        correct = (predicted == y).sum().item()  # 计算正确预测的样本数
    accuracy = correct / test_sample_num
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy

def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 类别数
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        logits = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        _, predicted = torch.max(logits, 1)  # 获取预测类别
    for vec, pred in zip(input_vec, predicted):
        print("输入：%s, 预测类别：%d" % (vec, pred.item()))

if __name__ == "__main__":
    main()
    # test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.bin", test_vec)
