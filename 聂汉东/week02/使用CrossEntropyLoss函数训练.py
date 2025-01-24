# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, input_size)  # 输出维度与输入维度相同
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    def forward(self, x, y=None):
        logits = self.linear(x)  # 计算 logits
        if y is not None:
            return self.loss(logits, y)  # 计算损失
        else:
            return torch.softmax(logits, dim=1)  # 返回概率分布

# 生成样本：最大值下标为类别
def build_sample(input_size):
    x = np.random.random(input_size)
    label = np.argmax(x)  # 最大值的下标作为类别标签
    return x, label

# 生成数据集
def build_dataset(total_sample_num, input_size):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample(input_size)
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 评估函数
def evaluate(model, input_size):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num, input_size)
    with torch.no_grad():
        y_pred = model(x).argmax(dim=1)  # 预测类别
        correct = (y_pred == y).sum().item()
    acc = correct / test_sample_num
    print(f"正确率：{acc:.2f}")
    return acc

def main():
    # 配置参数
    epoch_num = 100  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总样本数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率

    model = TorchModel(input_size)  # 初始化模型
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log = []
    train_x, train_y = build_dataset(train_sample, input_size)  # 生成训练数据

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算损失
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        print(f"第{epoch + 1}轮，平均loss: {np.mean(watch_loss):.4f}")
        acc = evaluate(model, input_size)  # 测试模型
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model.bin")

    # 绘制损失和准确率曲线
    plt.plot(range(len(log)), [l[0] for l in log], label="Accuracy")
    plt.plot(range(len(log)), [l[1] for l in log], label="Loss")
    plt.legend()
    plt.show()

# 使用训练好的模型预测
def predict(model_path, input_vec):
    input_size = len(input_vec[0])
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载权重
    model.eval()
    with torch.no_grad():
        predictions = model(torch.FloatTensor(input_vec)).argmax(dim=1)  # 输出类别
    for vec, pred in zip(input_vec, predictions):
        print(f"输入: {vec}, 预测类别: {pred.item()}")

if __name__ == "__main__":
    main()
    # test_vec = [[0.97889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
    #             [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
    #             [0.00797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
    #             [0.09349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]]
    # predict("model.bin", test_vec)
