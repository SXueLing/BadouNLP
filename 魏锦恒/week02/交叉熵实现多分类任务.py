import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，最大值所在的维度就为正样本

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes=5):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 5个类别
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(logits, y)  # 计算交叉熵损失
        else:
            return logits


# 生成一个样本，返回最大值所在维度的索引作为标签
def build_sample():
    x = np.random.random(5)
    label = np.argmax(x)  # 找到最大值的索引作为类别标签
    return x, label


# 随机生成一批样本
# 样本的类别是最大值所在的维度
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])  # 标签是最大值所在维度的索引
    return torch.FloatTensor(X), torch.LongTensor(Y).squeeze()  # 标签为类别索引


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        logits = model(x)  # 模型预测
        y_pred = torch.argmax(logits, dim=1)  # 取logits最大值的索引作为预测结果
        correct = (y_pred == y).sum().item()  # 计算正确预测的数量
        wrong = test_sample_num - correct
    print(f"正确预测个数：{correct}, 正确率：{correct / (correct + wrong):.6f}")
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 40  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 类别数目
    learning_rate = 0.001  # 学习率

    # 建立模型
    model = TorchModel(input_size, num_classes)
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
            loss = model(x, y)  # 计算loss
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


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        logits = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        y_pred = torch.argmax(logits, dim=1)  # 预测类别
        for vec, pred in zip(input_vec, y_pred):
            print(f"输入：{vec}, 预测类别：{pred.item()}, 概率值：{torch.softmax(logits, dim=1)[0][pred].item()}")

if __name__ == "__main__":
    main()
    # test_vec = [[0.97889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
    #             [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
    #             [0.00797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
    #             [0.09349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]]
    # predict("model.bin", test_vec)

