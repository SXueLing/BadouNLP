import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# 定义模型
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, input_size)  # 输出为input_size，分类时直接输出五个类别
        self.loss = nn.CrossEntropyLoss()  # 使用CrossEntropyLoss来做多分类

    # 前向传播
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size)
        if y is not None:
            return self.loss(x, y.squeeze().long())  # 计算多类交叉熵损失
        else:
            return x  # 预测值


# 随机生成一个5维向量，返回最大值所在的索引和对应的向量
def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)
    return x, max_index  # 生成向量和对应的类别标签（最大值索引）


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        _, predicted = torch.max(y_pred, 1)  # 选择最大输出作为预测类别
        correct = (predicted == y.squeeze()).sum().item()  # 计算预测正确的个数
        wrong = test_sample_num - correct
    print(f"正确预测个数：{correct}, 正确率：{correct / test_sample_num:.6f}")
    return correct / test_sample_num


# 主函数
def main():
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率

    model = TorchModel(input_size)
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
    print(log)

    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    model.eval()  # 测试模式
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    _, predicted = torch.max(result, 1)  # 获取最大值的索引
    for vec, res in zip(input_vec, predicted):
        print(f"输入：{vec}, 预测类别：{int(res)}, 概率值：{result[0][int(res)].item()}")  # 打印结果


if __name__ == "__main__":
    main()
