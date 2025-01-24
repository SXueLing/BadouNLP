# -*- coding = utf-8 -*-
# @Time : 2024-11-27 15:00
# @Author : 川
# @File : TorchTest(1).py
# @Software : PyCharm


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层，输出5个类别的得分
        self.activation = torch.softmax  # softmax归一化函数
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失

    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)
        y_pred = self.activation(x, dim=1)  # (batch_size, 5) -> (batch_size, 5)
        if y is not None:
            return self.loss(x, y.long())  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果

# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，最大的数字在哪维就属于哪一类
def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)
    return x, max_index

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
def evaluate(model, x, y):
    model.eval()
    class_counts = [0] * 5  # 初始化每个类别的计数器
    correct_counts = [0] * 5  # 初始化每个类别正确预测的计数器
    with torch.no_grad():
        y_pred = model(x)
        _, predicted = torch.max(y_pred, 1)
        for i in range(len(y)):
            class_counts[y[i]] += 1  # 统计每个类别的样本数量
            if predicted[i] == y[i]:
                correct_counts[y[i]] += 1  # 统计每个类别正确预测的样本数量
    accuracy = (predicted == y).sum().item() / len(y)  # 计算总准确率
    return accuracy, class_counts, correct_counts

def main():
    # 配置参数
    epoch_num = 50  # 训练轮数
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
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        acc, class_counts, correct_counts = evaluate(model, train_x, train_y)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print("总准确率：%f" % acc)
        print("每个类别的样本数量和正确预测数量：")
        for i in range(5):
            print("类别 %d: 样本数量 %d, 正确预测数量 %d" % (i, class_counts[i], correct_counts[i], ))
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
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model(torch.FloatTensor(input_vec))  # 模型预测
    _, predicted = torch.max(result, 1)
    for vec, pred in zip(input_vec, predicted):
        print("输入：%s, 预测类别：%d" % (vec, pred.item()))  # 打印结果

if __name__ == "__main__":
    main()
