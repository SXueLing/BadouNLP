# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as f
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，随机向量最大的数字在哪维就属于哪一类

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        # self.activation = torch.softmax  #去掉，实际cross_entropy内部会有算，如需要可以打印出来
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)
        # print(x)
        # print(y)
        # print(f.softmax(y_pred))   # 打印softmax后的各分布
        # y_pred = self.activation(x, dim=0)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，最大值输出对应的索引
def build_sample():
    x = np.random.random(5)
    return x, np.argmax(x)


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个样本" % (test_sample_num))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if float(torch.argmax(y_p)) == float(y_t):  # 预测的最大值与实际的最大值的索引位置比较
                correct += 1  # 判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 30  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    # train_sample = 20  # 每轮训练总共训练的样本总数
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
            # print(loss)
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
    # print(log)
    # plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    # plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    # plt.legend()
    # plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载训练好的权重
    # print("222", model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        # print("预测结果111",result)
        # print("预测结果：",torch.argmax(result,dim=1))
        sj_relute = np.argmax(input_vec, axis=1)
        # print("7777:",sj_relute)
        for vec, res, s in zip(input_vec, torch.argmax(result, dim=1), sj_relute):
            print("输入：%s, 预测类别：%d, 实际值：%d" % (str(vec), round(float(res)), s))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.6732, 0.3792, 0.0508, 0.3146, 0.2503],
                [0.5203, 0.9488, 0.3450, 0.6315, 0.5111],
                [0.6664, 0.7205, 0.4450, 0.0014, 0.4913],
                [0.7599, 0.1990, 0.2312, 0.6218, 0.1590],
                [0.2114, 0.1012, 0.0739, 0.9461, 0.9596],
                [0.4765, 0.0461, 0.9819, 0.6689, 0.5310],
                [0.7080, 0.7598, 0.4951, 0.8509, 0.0413],
                [0.9174, 0.9922, 0.9776, 0.3784, 0.5352],
                [0.7379, 0.5530, 0.2728, 0.7461, 0.1990],
                [0.5042, 0.5384, 0.9884, 0.4652, 0.1491]]
    predict("model.bin", test_vec)

