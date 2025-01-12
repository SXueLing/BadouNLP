import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的多分类(机器学习)任务
规律：x是一个5维向量，按向量中最大的数字在哪一维就属于哪一类（共五类）
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 修改线性层输出维度为类别数
        self.activation = nn.Softmax(dim=1)  # 使用Softmax激活函数，将输出转换为概率分布，dim=1表示按行进行
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数用于多分类任务

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        y_pred = self.activation(x)  # (batch_size, num_classes) -> (batch_size, num_classes)，转换为概率分布
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算交叉熵损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本，样本的生成方法，代表了我们要学习的新规律
# 随机生成一个5维向量，找到最大数字所在维度作为类别（0 - 4）
def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)  # 找到最大元素的索引，即类别
    return x, max_index


# 随机生成一批样本
# 均匀生成各类别的样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 注意标签转为LongTensor类型，符合交叉熵要求

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本次预测集中各类别样本数量:", np.bincount(y.squeeze().numpy()))  # 统计各类别样本数量
    correct = 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        _, predicted = torch.max(y_pred, 1)  # 获取预测的类别
        correct += (predicted == y.squeeze()).sum().item()
    accuracy = correct / test_sample_num
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 类别数量
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
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


if __name__ == "__main__":
    main()
