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

作业要求：改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。

数据样本
x数组中有多个样本（数组），一个样本中随机生成五个值
对应真实值为：样本中最大值的索引
"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, input_size)  # 线性层
        self.linear2 = nn.Linear(input_size, 5)  # 线性层
        # self.activation = torch.sigmoid  # sigmoid归一化函数
        # self.loss = nn.functional.mse_loss  # loss函数采用均方差损失
        self.loss = nn.CrossEntropyLoss()

    
    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        y_pred = self.linear2(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
def build_sample():
    x = np.random.random(5)
    # x = [round(random.random(), 4) for i in range(5)]
    # x = [random.randint(1, 10000) for i in range(5)]
    return x, np.argmax(x)


def build_dataset(total_sample_num):
    '''
        用来生成样本和真实值
    '''
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)
# print(build_dataset(10))
# 测试模型
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()  # 模型设置为测试模式
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)  # 生成预测数据，x 样本 ，  y 对应的真实值。
    # 样本可能有5类
    numof1 = numof2 = numof3 = numof4 = numof5 = 0
    for item in y:
        case = item.item()
        if case == 0:
            numof1 += 1
        elif case == 1:
            numof2 += 1
        elif case == 2:
            numof3 += 1
        elif case == 3:
            numof4 += 1
        elif case == 4:
            numof5 += 1

    print(f"本次测试中，一类样本有{numof1}个, 二类样本有{numof2}个，三类样本有{numof3}个, 四类样本有{numof4}个，五类样本有{numof5}个, 一共有{sum([numof1, numof2, numof3, numof4, numof5])}个")
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        # y_pred = model.forward(x)  # 模型预测 model.forward(x)
        # print("本次测试的x值为：", x, y_pred, y)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            # print("本次预测值y_p值为：", y_p, y_t)
            max_value, max_index = torch.max(y_p, dim=0)
            # print("max_index为：",max_index,"y_t为：", y_t, max_index==y_t)
            if max_index == y_t:
                correct += 1  # 样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 3000  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate =0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器（就是那条带学习率计算新权重的公式 w新 = w旧 - lr * 梯度）
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，真正项目中是读取训练集（就是生成数据）
    train_x, train_y = build_dataset(train_sample)

    # 训练过程：
    for epoch in range(epoch_num):
        model.train()  # 模型设置为训练模式（模型中网络层训练和预测时候表现不一样）
        watch_loss = []  # 保存每一轮的loss 方便我们自己查看
        for batch_index in range(train_sample // batch_size):
            # 取出每一个batch的样本和真实值
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            # 计算损失
            loss = model(x, y)  # 【重点】等效于 model.forward(x, y)  框架帮我们封装好了
            loss.backward()  # 计算梯度，即求导
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        if np.mean(watch_loss) < 0.05:
            print('-----------------------------------------------------------------训练停止')
            break;
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果（每一轮训练完做测试）
        # log.append([acc, float(np.mean(watch_loss))])
    # 训练完毕，保存模型
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
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        # input_vec 是样本
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        print("result为：", result)

    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d" % (vec, torch.max(res, dim=0)[1]))  # 打印结果


if __name__ == "__main__":
    a = 1
    main()
    test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.bin", test_vec)
