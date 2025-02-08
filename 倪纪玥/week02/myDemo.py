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

第二周作业
改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        output_size = 5
        self.linear = nn.Linear(input_size, output_size)  # 线性层 输入维度 输出维度为 5, 5个中最大的概率为就属于哪类
        # 移除激活函数，直接使用线性输出（因为 CrossEntropyLoss 交叉熵损失在内部会执行 Softmax 操作）
        self.loss = nn.CrossEntropyLoss()         # 损失函数为交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    # 前向计算
    def forward(self, x, y=None):
        # 线性层
        y_pred = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)
        # 不需要 激活层
        # y_pred = self.activation(x)  # (batch_size, 5) -> (batch_size, 5)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，五维随机向量最大的数字在哪维就属于哪一类
def build_sample():
    x = np.random.random(5)        # 生成一个包含5个随机数的向量
    y = np.argmax(x)               # 找到最大值的索引作为类别标签
    return x, y


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)     # 将 build_sample 返回的 y已经是向量了,所以不需要加 [],因此 Y.append([y]) 改为 Y.append(y)
    X = np.array(X)     # 使用 np.array()优化列表到张量的转换性能
    Y = np.array(Y)
    return torch.FloatTensor(X), torch.LongTensor(Y)   # 使用 torch.LongTensor(Y) 将标签转换为长整型张量，这是交叉熵损失函数所需的标签类型。


def evaluate(model):
    model.eval()  # 切换到评估模式
    test_sample_num = 500
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个类别样本" % test_sample_num)
    correct = 0
    with torch.no_grad():
        y_pred_logits = model(x)  # 获取模型的输出（logits）
        y_pred = torch.argmax(y_pred_logits, dim=1)  # 获取预测的类别索引
        correct = (y_pred == y).sum().item()  # 计算预测正确的数量
    total = y.size(0)
    accuracy = correct / total
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy


def main():
    # 配置参数 超参数
    epoch_num = 100  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器 Adam 类似于梯度下降sgd
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num): # 训练次数
        model.train()   # 把模型设置为训练的模式 某些网络层在训练和预测的时候的表现不一致的
        watch_loss = []
        # 循环批次输入数据
        for batch_index in range(train_sample // batch_size): # 取整除 - 返回商的整数部分（向下取整）
            # 每次送进去 batch_size 个数据进行训练,优化模型
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model(x, y) 等效于 model.forward(x,y)
            loss.backward()  # 计算梯度, 根据损失函数进行计算, 完成所有的线性层的计算
            optim.step()  # 更新权重 adam优化进行更新
            optim.zero_grad()  # 梯度归零 每个batch计算完成需要将梯度归0
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "myModel.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线 准确率图像
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线 loss的图像
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    model.eval()  # 切换到评估模式
    with torch.no_grad():  # 不计算梯度
        input_tensor = torch.FloatTensor(input_vec)
        output_logits = model(input_tensor)  # 模型预测，得到输出的 logits
        predicted_classes = torch.argmax(output_logits, dim=1)  # 获取预测的类别索引
        probabilities = torch.softmax(output_logits, dim=1)  # 将 logits 转化为概率分布

    # 打印结果
    for vec, pred_class, prob in zip(input_vec, predicted_classes, probabilities):
        print("输入：%s, 预测类别：%d, 概率分布：%s" % (vec, pred_class.item(), prob.numpy()))


if __name__ == "__main__":
    main()
    test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("myModel.pt", test_vec)

    # 0 2 1 2
