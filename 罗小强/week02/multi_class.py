# -*- coding: utf-8 -*-
"""
multi_class.py
描述: 
作者: TomLuo 21429503@qq.com
日期: 11/26/2024
版本: 1.0
"""
# coding:utf8

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，哪一维的值最大，则认为属于哪一类
例如：x = [0.1, 0.2, 0.3, 0.4, 0.5]，那个数最大，则属于第几类
输出则为 [0, 0, 0, 0, 1]
如输出为 [1, 0, 0, 0, 0]，则属于第1类
如输出为 [0, 1, 0, 0, 0]，则属于第2类
如输出为 [0, 0, 1, 0, 0]，则属于第3类
如输出为 [0, 0, 0, 1, 0]，则属于第4类
如输出为 [0, 0, 0, 0, 1]，则属于第5类
"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.activation = nn.Softmax(dim=1)  # softmax激活函数
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y.argmax(dim=1))  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


def convert_to_one_hot(x):
    max_idx = np.argmax(x)
    one_hot = np.zeros_like(x)
    one_hot[max_idx] = 1
    return one_hot


def build_sample():
    x = np.random.random(5)
    return x, convert_to_one_hot(x)


def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)


def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # 计算每个类别的样本数量
    class_counts = [0] * 5
    for x_t, y_t in zip(x, y):
        class_counts[np.argmax(y_t)] += 1
    print("各类别样本数量：", class_counts)  # 预测结果各类别样本数量
    # 预测准确率
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == torch.argmax(y_t):
                correct += 1  # 判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
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


class TorchModel(nn.Module):
    """
    TorchModel类继承自nn.Module，用于构建神经网络模型。

    Attributes:
        input_size (int): 输入数据的特征维度。
        linear: 线性层，将输入数据映射到5维输出。
        activation: Softmax激活函数，用于将线性层的输出转换为概率分布。
        loss: 交叉熵损失函数，用于计算模型预测与真实标签之间的差异。
    """

    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.activation = nn.Softmax(dim=1)  # softmax激活函数
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        """
        TorchModel的前向传播方法。

        Parameters:
            x (torch.Tensor): 输入数据，形状为(batch_size, input_size)。
            y (torch.Tensor, optional): 真实标签，形状为(batch_size, num_classes)。默认为None。

        Returns:
            torch.Tensor: 如果提供真实标签y，则返回loss值；否则返回预测值y_pred。
        """
        x = self.linear(x)  # 线性变换
        y_pred = self.activation(x)  # 激活函数
        if y is not None:
            return self.loss(y_pred, y.argmax(dim=1))  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


def convert_to_one_hot(x):
    """
    将输入向量x转换为one-hot编码。

    Parameters:
        x (np.ndarray): 输入向量。

    Returns:
        np.ndarray: 转换后的one-hot编码向量。
    """
    max_idx = np.argmax(x)
    one_hot = np.zeros_like(x)
    one_hot[max_idx] = 1
    return one_hot


def build_sample():
    """
    构建单个样本数据。

    Returns:
        tuple: 输入样本x和对应的one-hot编码标签y。
    """
    x = np.random.random(5)
    return x, convert_to_one_hot(x)


def build_dataset(total_sample_num):
    """
    构建数据集。

    Parameters:
        total_sample_num (int): 数据集中的样本总数。

    Returns:
        tuple: 输入数据集X和对应的标签数据集Y，两者均为torch.FloatTensor类型。
    """
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)


def evaluate(model):
    """
    评估模型性能。

    Parameters:
        model (TorchModel): 需要评估的模型。

    Returns:
        float: 模型在测试集上的准确率。
    """
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # 计算每个类别的样本数量
    positive_samples = y.sum(dim=0)  # 每个类别的正样本数量
    negative_samples = test_sample_num - positive_samples  # 每个类别的负样本数量
    print("本次预测集中共有正样本：", positive_samples.numpy(), "负样本：", negative_samples.numpy())
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == torch.argmax(y_t):
                correct += 1  # 判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    """
    主函数，用于训练模型并进行评估。
    """
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
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
    """
    使用训练好的模型进行预测。

    Parameters:
        model_path (str): 模型权重文件的路径。
        input_vec (list): 输入向量列表，每个向量的维度必须与模型输入维度一致。
    """
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        # 获取最大概率对应的类别
        predicted_class = torch.argmax(res).item()
        # 获取最大概率值
        probability = res[predicted_class].item()
        print("输入：%s, 预测类别：%d, 概率值：%f, 最大概率对应的值：%f" % (
            vec, predicted_class, probability, vec[predicted_class]))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.86889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
                [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.00797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.14349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]]
    predict("model.bin", test_vec)


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        # 获取最大概率对应的类别
        predicted_class = torch.argmax(res).item()
        # 获取最大概率值
        probability = res[predicted_class].item()
        print("输入：%s, 预测类别：%d, 概率值：%f, 最大概率对应的值：%f" % (
            vec, predicted_class, probability, vec[predicted_class]))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.86889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
                [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.00797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.14349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]]
    predict("model.bin", test_vec)
