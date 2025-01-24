import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt


# 修改后的神经网络模型
class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失

    def forward(self, x, y=None):
        logits = self.linear(x)  # 线性变换输出 logits
        if y is not None:
            return self.loss(logits, y)  # 计算交叉熵损失
        else:
            return logits  # 返回 logits 供预测


# 修改后的样本生成函数
# 随机生成一个样本：5维向量和最大值的索引作为类别
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)  # 返回最大值的索引作为类别标签
    return x, y

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
    print(f"测试样本数：{test_sample_num}")
    correct = 0
    with torch.no_grad():
        logits = model(x)  # 模型预测
        predictions = torch.argmax(logits, dim=1)  # 获取预测类别
        correct = (predictions == y).sum().item()
    accuracy = correct / test_sample_num
    print(f"正确预测个数：{correct}, 正确率：{accuracy:.4f}")
    return accuracy


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总样本数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 分类类别数
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
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print(f"第{epoch + 1}轮平均loss: {np.mean(watch_loss):.4f}")
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model.bin")




if __name__ == "__main__":
    main()
