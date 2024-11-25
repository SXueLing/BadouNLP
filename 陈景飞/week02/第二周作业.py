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
规律：x是一个5维向量，实现一个5分类的任务，向量中哪个数最大，就代表第几类。使用交叉熵。

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# python找出数组中最大的数，及其下标
def get_max_value_and_index(x):
    max_value = x[0]
    max_index = 0
    for index, value in enumerate(x):
        if value > max_value:
            max_value = value
            max_index = index
    return max_value, max_index


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，，向量中哪个数最大，就代表第几类。使用交叉熵。x=[1,2,3,4,5] ---> y=[0,0,0,0,1]
def build_sample():
    x = np.random.random(5)
    max_value, max_index = get_max_value_and_index(x)
    y_true_array = np.zeros(x.size)
    y_true_array[max_index] = 1
    return x, y_true_array


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # print("x====", x)
    # print("y====", y)

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            # print("11111", x)
            # print("22222", y_p)
            # print("33333", y_t)
            y_p = y_p.numpy()
            y_t = y_t.numpy()
            # print("44444", y_p)
            # print("5555", y_t)
            indices1 = np.where(y_t == 1.0)[0]  # [0] 是因为 numpy.where 返回的是一个元组，对于一维数组我们关心第一个元素
            # print("indices1=",indices1)
            max_value, max_index = get_max_value_and_index(y_p)
            indices2 = max_index
            if indices1 == indices2:
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
            # print("y_p====", y_p)
        print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
        return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 200  # 训练轮数
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
    torch.save(model.state_dict(), "homework2.bin")
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
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        np.set_printoptions(precision=5, suppress=True)
        print("result===========")
        print(result.numpy())
    for vec, res in zip(input_vec, result):
        max_value, max_index = get_max_value_and_index(res)
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, max_index + 1, max_value))  # 打印结果


if __name__ == "__main__":
    main()
    # test_vec = [[0.97889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
    #             [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
    #             [0.00797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
    #             [0.09349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894],
    #             [1.00000000, 2.00000000, 3.00000000, 4.00000000, 5.00000000],
    #             [5.00000000, 4.00000000, 3.00000000, 2.00000000, 1.00000000]]
    # predict("homework2.bin", test_vec)
