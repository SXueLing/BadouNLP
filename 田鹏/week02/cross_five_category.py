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
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.activation = torch.sigmoid
        self.loss = nn.CrossEntropyLoss()  # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):  # 只有输入的时候，输出预测值，输入和预测值都有的时候，输出损失值
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def build_sample():
    x = np.random.random(5)  # 随机生成5维向量
    if list(x).index(max(x)) == 0:
        return x, 0  # 如果第一个大于第五个值，就返回1，否则就返回0
    elif list(x).index(max(x)) == 1:
        return x, 1
    elif list(x).index(max(x)) == 2:
        return x, 2
    elif list(x).index(max(x)) == 3:
        return x, 3
    else:
        return x, 4

def build_dataset(total_sample_num):  # 指定生成样本数量
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 标签是标量，为了能让pytorch能够明白每一个标量都是一个样本，需要括号，如果是一个向量就不需要
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()  # 调整模型的状态，告诉他现在不是训练模式
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)  # 生成100个样本
    print("本次预测集中共有%d个一类，%d个二类,%d个三类,%d个四类,%d个五类" % (list(y).count(0),
                                                                    list(y).count(1),
                                                                    list(y).count(3),
                                                                    list(y).count(4),
                                                                    list(y).count(5)))
    correct, wrong = 0, 0  # 正确，错误有多少
    with torch.no_grad():  # 现在不用算梯度
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比 y_p预测值  y_t 真实值
            y_pp = list(y_p).index(max(y_p))
            if y_pp == int(y_t):  # y_pp == y_t : #float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1  # 负样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 训练主流程
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
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   # Adam类似SGD，梯度下降法的优化器
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):  # 类似  for epoch in range(1000) 循环训练轮数
        model.train()  # 把模型设置成训练的模式
        watch_loss = []  # 存储loss的东西
        for batch_index in range(train_sample // batch_size):     # 类似for x, y_true in zip(X, Y) 循环训练样本
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]  # 每一轮取出0-20个数据进行训练
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss 等效于 model.forward(x,y)
            loss.backward()  # 计算梯度  和上面那个是固定写法
            optim.step()  # 更新权重 类似 w1 = w1 - lr * grad_w1/batch_size
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "week2_job.bin")  # 保存模型的权重
    # 画图
    print('正确率和损失函数',log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):  #模型由权重决定的
    input_size = 5
    model = TorchModel(input_size)  # 如果不进行加载，则会初始化
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        cate_index = list(res.tolist()).index(max(res.tolist()))
        print("输入：%s, 预测类别：%d, 概率值：%s" % (vec, cate_index, res.tolist()))  # 打印结果
        # print("输入：%s,  概率值：%f" % (vec, res))


if __name__ == "__main__":
    main()
    test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("week2_job.bin", test_vec)  # 模型的文件名字
