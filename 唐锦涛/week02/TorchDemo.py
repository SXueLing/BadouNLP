# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，五维随机向量最大的数字在哪维就属于哪一类

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.activation = torch.softmax  # Softmax归一化函数
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x, -1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果一个值为最大值，则输出该值所在的位置
def build_sample():
    x = np.random.random(5)
    y = x.max()
    for i in range(5):
        if x[i] == y:
            return x, i


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append([x])
        Y.append([y])
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        for x_t, y_t in zip(x, y):  # 与真实标签进行对比
            y_pred = model(x_t)  # 模型预测 model.forward(x)
            # 拿到最大值
            max = y_pred[0].max()
            # print(max)
            y_p = None
            for i in range(5):
                if y_pred[0][i] == max:
                    # 拿到最大值的下标
                    y_p = i
            # 当最大值的小标和准确值相等时，则为正确
            if y_p == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.121  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample):
            x = train_x[batch_index]
            y = train_y[batch_index]
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
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        # print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果
        max = res.max()
        type = None
        for i in range(5):
            if res[i] == max:
                type = i
        print("输入：%s, 输出：%s, 种类：%s" % (vec, res, type))  # 打印结果


if __name__ == "__main__":
    # main()
    test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894],
                [0.20254402,0.138746,0.09637442,0.65043198,0.20928234],
                [0.41051344,0.07225582,0.35347485,0.77086436,0.57336498],
                [0.53106459,0.79182282,0.91852906,0.76992054,0.36576879],
                [0.21640893,0.97978838,0.06092612,0.28999622,0.0510368]]
    predict("model.bin", test_vec)
