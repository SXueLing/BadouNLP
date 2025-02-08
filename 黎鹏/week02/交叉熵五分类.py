
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

# 实现一个多分类任务：五维随机向量，最大值所在的维度即为类别

class TorchModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.linear(x)
        if y is not None:
            return self.loss(x, y)  # 计算交叉熵损失
        else:
            return x  # 输出预测值


# 生成一个样本，五维向量，最大值所在维度作为标签
def build_sample():
    x = np.random.random(5)
    target = np.argmax(x)
    return x, target

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 标签需要是 Long 类型


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        _, predicted = torch.max(y_pred, 1)  # 获取预测值的最大值的索引（类别）
        correct = (predicted == y).sum().item()  # 计算正确预测的数量
    print(f"本次预测集中共有{test_sample_num}个样本，正确预测个数：{correct}, 正确率：{correct / test_sample_num:.4f}")
    return correct / test_sample_num


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    output_size = 5  # 输出类别数（5类）
    learning_rate = 0.001  # 学习率

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
            loss = model(x, y)  # 计算损失
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss: {np.mean(watch_loss):.6f}")
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
    model = TorchModel(input_size, 5)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        _, predicted = torch.max(result, 1)  # 获取预测类别

    for vec, res in zip(input_vec, predicted):
        print(f"输入：{vec}, 预测类别：{res.item()}")


if __name__ == "__main__":
    main()
    # 测试向量
    # test_vec = [[0.1, 0.2, 0.3, 0.4, 0.5],
    #             [0.7, 0.1, 0.3, 0.2, 0.6],
    #             [0.2, 0.8, 0.3, 0.4, 0.9],
    #             [0.6, 0.6, 0.6, 0.7, 0.8]]
    # predict("model.bin", test_vec)
