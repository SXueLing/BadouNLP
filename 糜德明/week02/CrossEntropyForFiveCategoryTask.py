import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律（机器学习）任务
规律：改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。
"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层

    # 当输入真实的标签，返回loss值，无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)
        if y is not None:
            # 如果给定标签，计算损失
            loss = nn.functional.cross_entropy(x, y)  # 计算交叉熵损失
            return loss
        else:
            return x  # 如果没有真实标签，返回 logits

# 生成一个样本，样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本

def build_sample():
    x = np.random.random(5)
    y = np.zeros(5)
    y[np.argmax(x)] = 1
    return x, y,

# 随机生成一批样本
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
    model.eval()  # 设置模型为评估模式
    test_sample = 100
    x, y = build_dataset(test_sample)
    # print("========\n测试集共有%d个正样本，%d个负样本" % (sum(y), test_sample-sum(y)))
    current, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        y_pred_classes = torch.argmax(y_pred, dim=1)
        # 获取预测类别索引
        for true, pred in zip(y, y_pred_classes):
            index = torch.nonzero(true == 1).squeeze()
            if index == pred:
                current += 1
            else:
                wrong += 1
    print("测试集共有%d个样本，预测正确%d个，错误%d个，准确率为%f" % (test_sample, current, wrong, current / test_sample))
    return current / (current + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器（之前用的是梯度下降SGD）
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()  # 设置模型为训练模式
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 获取一个batch的数据
            batch_x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            batch_y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(batch_x, batch_y)  # 计算loss 等价于 model.forward(batch_x, batch_y)
            loss.backward()  # 反向传播,计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 清空梯度
            watch_loss.append(loss.item())
        print("========\n第%d轮平均loss：%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    # 保存模型
    torch.save(model.state_dict(), "model2.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 成功率
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 损失函数
    plt.legend()
    plt.show()
    return

def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
        y_pred_classes = torch.argmax(result, dim=1)
    for vec, res in zip(input_vec, y_pred_classes):
        print("输入：%d, 预测的结果：%d" %(np.argmax(vec), res))

if __name__ == "__main__":
    main()
    # predict("model2.bin", [[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1], [1,2,4,2,1]])
