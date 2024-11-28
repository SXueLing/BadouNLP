# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
多分类任务：给定一个5维向量，判断哪个维度的值最大，将其作为类别（0, 1, 2, 3, 4）。
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  # 线性层
        self.loss = nn.CrossEntropyLoss()  # 多分类任务使用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, input_size) -> (batch_size, output_size)
        if y is not None:
            return self.loss(logits, y)  # 计算损失
        else:
            return logits  # 输出logits


# 生成一个样本，类别为最大值所在的索引
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)  # 最大值的索引即为类别
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))  # 注意类别标签类型为LongTensor


# 测试代码
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        logits = model(x)  # 模型预测
        y_pred = torch.argmax(logits, dim=1)  # 获取预测的类别
        correct = (y_pred == y).sum().item()
        wrong = test_sample_num - correct
    accuracy = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy


def main():
    # 配置参数
    epoch_num = 100  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    output_size = 5  # 输出类别数
    learning_rate = 0.005  # 学习率
    # 建立模型
    model = TorchModel(input_size, output_size)
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
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model_multiclass.bin")
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
    model = TorchModel(input_size, output_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        logits = model(torch.FloatTensor(input_vec))  # 模型预测
        predictions = torch.argmax(logits, dim=1)  # 获取预测类别
    for vec, pred in zip(input_vec, predictions):
        print("输入：%s, 预测类别：%d" % (vec, pred))  # 打印结果


if __name__ == "__main__":
    main()
    # test_vec = [[0.978, 0.152, 0.310, 0.035, 0.889],
    #             [0.749, 0.552, 0.957, 0.955, 0.848],
    #             [0.007, 0.674, 0.136, 0.346, 0.198],
    #             [0.093, 0.594, 0.925, 0.415, 0.135]]
    # predict("model_multiclass.bin", test_vec)
