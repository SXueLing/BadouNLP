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
作业：用交叉熵实现一个多分类任务，五维随机向量中最大数字所在维度即目标类别

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失

    # 当输入真实值标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None): #串联各个层
        y_pred = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，最大数字所在维度即目标类别
def build_sample():
    x = np.random.random(5)
    max_index = 0
    max_value = x[0]
    for i in range(5):
        if x[i] > max_value:
            max_index = i
            max_value = x[i]
    return x, max_index

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y) #向量转变成张量

# 测试代码，非模型训练
# 用来测试每轮模型的准确率，评价当前训练效果
def evaluate(model):
    model.eval() #将model模型设置到测试模式
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad(): #此时不用梯度，使模型计算更快
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):
            class_pred, max_pred = max((i, v) for i, v in enumerate(y_p))
            class_ture = int(y_t)
            if class_pred == class_ture:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

# 训练的主流程***
def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数（这里250个batch批次为一轮，每个batch批次为20个数据，总共20轮）
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size) #建立名为model的线性层模型
    # 选择并配置优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate) #Adam优化器，括号内告知选择的模型是model，即对model模型内的权重进行更新
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程（以上都是前期准备）
    for epoch in range(epoch_num):
        model.train() #将model模型设置到训练模式
        watch_loss = [] #存储每轮的loss函数来看
        for batch_index in range(train_sample // batch_size):   #遍历每一个批次batch（每个batch有20个数据）
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size] #train_x的切片
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size] #train_y的切片
            # 以下是整个训练过程的核心步骤
            loss = model(x, y)  # 计算loss 等效于 loss = model.forward(x,y)
            loss.backward()  # 计算梯度即计算loss函数每个权重的偏导数，这是固定写法，loss函数内置有梯度的计算
            optim.step()  # 优化器根据梯度来更新权重
            optim.zero_grad()  # 梯度归零，计算完一个batch后梯度归零，再计算下一个batch
            # 以上四步是核心训练过程：损失函数，梯度计算，权重更新，梯度归零
            watch_loss.append(loss.item()) #显示loss函数的数值
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pt") #bin是文件名字，任意都可以

    # 画图（不是模型训练的必要部分，可以省略）
    # print(log)
    # plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    # plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    # plt.legend()
    # plt.show()
    # return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, torch.argmax(res), res))  # 打印结果


if __name__ == "__main__":
    #main()
    test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.pt", test_vec)
