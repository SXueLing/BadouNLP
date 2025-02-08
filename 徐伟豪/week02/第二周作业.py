import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层，输出为5类
        self.activation = nn.Softmax(dim=1)  # 对于多类分类使用Softmax函数

    def forward(self, x):
        x = self.linear(x)  # 通过线性层处理输入
        return self.activation(x)  # 应用Softmax函数


def build_sample():
    x = np.random.random(5)  # 生成一个5维随机向量
    return x, np.argmax(x)  # 返回随机向量和最大值对应的索引作为类别标签


def build_dataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample()  # 生成样本
        X.append(x)
        Y.append(y)  # 记录类别标签
    return torch.FloatTensor(X), torch.LongTensor(Y)  # X为FloatTensor，Y为LongTensor用于交叉熵损失


def evaluate(model):
    model.eval()  # 设置模型为评估模式
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)  # 生成测试样本
    correct = 0
    with torch.no_grad():  # 不计算梯度
        y_pred = model(x)  # 获取模型预测结果
        predicted_classes = torch.argmax(y_pred, dim=1)  # 获取预测类别
        correct = (predicted_classes == y).sum().item()  # 计算正确预测的数量
    accuracy = correct / test_sample_num  # 计算准确率
    print(f"正确预测个数：{correct}, 正确率：{accuracy:.6f}")
    return accuracy


def main():
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每个批次的样本数量
    train_sample = 5000  # 训练总样本数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率

    model = TorchModel(input_size)  # 创建模型
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 选择优化器
    log = []

    train_x, train_y = build_dataset(train_sample)  # 生成训练集

    for epoch in range(epoch_num):
        model.train()  # 设置模型为训练模式
        watch_loss = []  # 用于记录每轮的损失
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]  # 当前批次的输入
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]  # 当前批次的标签
            optim.zero_grad()  # 梯度归零

            y_pred = model(x)  # 获取预测结果
            loss = nn.CrossEntropyLoss()(y_pred, y)  # 计算交叉熵损失
            loss.backward()  # 反向传播
            optim.step()  # 更新权重

            watch_loss.append(loss.item())  # 记录损失

        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss):.6f}")
        acc = evaluate(model)  # 测试当前模型
        log.append([acc, float(np.mean(watch_loss))])  # 记录准确率和损失

    torch.save(model.state_dict(), "model.bin")  # 保存模型
    print(log)  # 打印训练日志
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 绘制准确率曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 绘制损失曲线
    plt.legend()
    plt.show()


def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)  # 创建模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 不计算梯度
        result = model(torch.FloatTensor(input_vec))  # 获取模型预测结果
    predicted_classes = torch.argmax(result, dim=1)  # 获取预测类别
    for vec, res in zip(input_vec, result):
        print(f"输入：{vec}, 预测类别：{predicted_classes.item()}, 概率值：{res}")  # 打印输入和预测结果


if __name__ == "__main__":
    main()  # 运行主程序
