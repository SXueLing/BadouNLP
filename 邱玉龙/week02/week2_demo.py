#  -*- coding: utf-8 -*-
"""
Author: loong
Time: 2024/11/26 0:06
File: week2_demo.py
Software: PyCharm
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        # # 隐含层1，输入维度是input_size，输出维度为128
        # self.hidden1 = nn.Linear(input_size, 128)
        # self.activation1 = nn.ReLU()  # 使用ReLU激活函数
        # # 隐含层2，输入维度128，输出维度为64
        # self.hidden2 = nn.Linear(128, 64)
        # self.activation2 = nn.ReLU()
        # # 输出层，输出3个类别
        # self.output_layer = nn.Linear(64, 3)
        self.linear = nn.Linear(input_size, 3)  # 线性层
        self.activation = torch.softmax   # softmax 激活函数
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵


    def forward(self, x, y=None):
        # 通过隐含层1
        # x = self.hidden1(x)
        # x = self.activation1(x)
        # # 通过隐含层2
        # x = self.hidden2(x)
        # x = self.activation2(x)
        # # 通过输出层
        # y_pred = self.output_layer(x)
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        y_pred = self.activation(x, dim=1)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果
# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，分别代表 温度，湿度，风速，气压，降水量
# 假设天气分类最具相关性的特征 温度，风速，降水量
def build_sample():
    x = np.random.random(5)
    # 防止 x 为 0 加个最小值
    probabilities = np.array([
        x[0] / (x.sum() + 1e-6),
        x[2] / (x.sum() + 1e-6),
        x[4] / (x.sum() + 1e-6)
    ])
    # 概率归一化 保证概率总和为 1
    probabilities /= probabilities.sum()
    # print(probabilities)
    # [0.18658355 0.21834961 0.59506684]
    # 说明 类别 0 选中概率 18%  类别1 概率 22%  类别3 概率60%
    label = np.random.choice([0, 1, 2], p=probabilities)

    one_hot_label = [1 if i == label else 0 for i in range(3)]
    # 如果 one_hot_label=> [1,0,0] 那么x 为 类别0 也就是晴天样本
    return x, one_hot_label

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
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)

        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            # print("****************")
            # print(y_p,y_t)
            if torch.argmax(y_p) == torch.argmax(y_t):
                # print("true")
                correct +=1
            else:
                wrong += 1
                # print("false")
            # print("****************")

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 50  # 训练轮数
    batch_size = 15  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    log = []

    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_x, train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        # for index in range(0,train_sample,batch_size):
        #     sub_x = train_x[index :index + batch_size]
        #     sub_y = train_y[index : index +  batch_size]
        for batch_index in range(train_sample // batch_size):
            sub_x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            sub_y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(sub_x,sub_y) #计算损失函数
            loss.backward() #计算梯度
            optim.step() #更新权重
            optim.zero_grad()  # 梯度归0
            watch_loss.append(loss.item())
        acc = evaluate(model)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
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


if __name__ == '__main__':
    main()
