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

改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。

"""


class TorchModel(nn.Module):
    def __init__(self, input_size,output_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size,bias=True)  # 线性层  值在（0，1）
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):# 不传x,y 正常前向传播；传x,y 计算loss值
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        y_pred = nn.functional.softmax(x,dim=1)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def build_sample():
    x = np.random.random(5)
    max_in_x = 0
    index = 0
    for i in range(len(x)):
        if max_in_x < x[i]:
            max_in_x = x[i]
            index = i
    y = []
    for i in range(len(x)):
        if i != index:
            y.append(0)
        else:
            y.append(1)
    return x, y


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)# 为了pytorch的框架的使用方式
    return torch.FloatTensor(X), torch.FloatTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    #print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            index_pred, index_true = 0, 0
            max_pred, max_true = 0, 0
            for i in range(len(y_p)):
                y_each = y_p[i]
                if max_pred < y_each:
                    max_pred = y_each
                    index_pred = i
            for i in range(len(y_t)):
                if max_true < y_t[i]:
                    max_true = y_t[i]
                    index_true = i

            if index_pred != index_true:
                wrong += 1
            else:
                correct += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 训练的主流程
def main():
    # 配置参数
    epoch_num = 60  # 训练轮数
    batch_size = 10  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    output_size=5
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size,output_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)# 目前adam更常见，效果一般更好
    # optim = torch.optim.SGD(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):  # 循环取出训练数据
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)# 框架封装的简易方法
            loss.backward()  ## 计算梯度 使用自带原装的loss就有backward()
            optim.step()  ## 更新权重
            optim.zero_grad()  # 梯度归零 每个batch都需要梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "ce_model.bin")
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
    output_size = 5
    model = TorchModel(input_size,output_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    # for vec, res in zip(input_vec, result):
        # print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果
        # print(result)
        for vec, res in zip(input_vec, result):
            print("输入：%s, 预测类别：%d, 概率值：%f" % (vec,vec.index(max(vec)),max(vec)))

if __name__ == "__main__":
    main() # 训练模型
    test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843], # 使用训练好的模型文件进行预测
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("ce_model.bin", test_vec)
