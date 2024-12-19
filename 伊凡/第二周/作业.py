# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层
        self.loss_fn = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失

    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            return self.loss_fn(logits, y)  # 预测值和真实值计算损失
        else:
            return logits  # 输出预测结果


# 生成一个样本
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)  # 标签是最大值的索引
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
def evaluate(model, test_x, test_y):
    model.eval()
    with torch.no_grad():
        logits = model(test_x)
        _, predicted = torch.max(logits, dim=1)
        correct = (predicted == test_y).sum().item()
        accuracy = correct / len(test_y)
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy

def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    num_classes = 5  # 类别数

    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log = []

    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    # 创建测试集
    test_x, test_y = build_dataset(100)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(0, train_sample, batch_size):
            x = train_x[batch_index:batch_index + batch_size]
            y = train_y[batch_index:batch_index + batch_size]

            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, test_x, test_y)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model.bin")

    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    model.eval()  # 测试模式

    with torch.no_grad():  # 不计算梯度
        input_tensor = torch.FloatTensor(input_vec)
        logits = model(input_tensor)  # 模型预测
        _, predicted = torch.max(logits, dim=1)  # 获取预测类别

    for vec, res in zip(input_vec, predicted):
        print("输入：%s, 预测类别：%d" % (vec, res.item()))  # 打印结果

if __name__ == "__main__":
    main()
    test_vec = [[0.97889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843]]
    test_vec.extend(np.random.rand(1000, 5).tolist())
    predict("model.bin", test_vec)
