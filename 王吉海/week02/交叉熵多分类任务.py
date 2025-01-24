import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt


# 搭建网络
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 5)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)  #激活函数
        self.loss = nn.CrossEntropyLoss()  #损失函数使用交叉熵

    def forward(self, x, y=None):
        x = self.linear1(x)
        y_pred = self.sigmoid(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


# 构建样本
def build_sample():
    x = np.random.random(5)
    if max(x) == x[0]:
        return x, [1, 0, 0, 0, 0]
    elif max(x) == x[1]:
        return x, [0, 1, 0, 0, 0]
    elif max(x) == x[2]:
        return x, [0, 0, 1, 0, 0]
    elif max(x) == x[3]:
        return x, [0, 0, 0, 1, 0]
    elif max(x) == x[4]:
        return x, [0, 0, 0, 0, 1]


# 生成样本
def build_dataset(num):
    X = []
    Y = []
    for i in range(num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# 测试模型
def evaluate(model):
    model.eval()
    test_sample = 100
    x, y = build_dataset(test_sample)
    my_list = [0, 0, 0, 0, 0]
    for var in y:
        my_list[0] += int(var[0])
        my_list[1] += int(var[1])
        my_list[2] += int(var[2])
        my_list[3] += int(var[3])
        my_list[4] += int(var[4])
    print("本次预测中5类数量为%s" % my_list)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if y_p[0] >= 0.5 and y_t[0] == 1:
                correct += 1
            elif y_p[1] >= 0.5 and y_t[1] == 1:
                correct += 1
            elif y_p[2] >= 0.5 and y_t[2] == 1:
                correct += 1
            elif y_p[3] >= 0.5 and y_t[3] == 1:
                correct += 1
            elif y_p[4] >= 0.5 and y_t[4] == 1:
                correct += 1
            else:
                wrong += 1
    print("正确个数：%d,错误个数：%d,正确率%f" % (correct, wrong, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 训练模型
def main():
    epochs = 40
    batch_size = 20
    learning_rate = 0.001
    train_sample = 10000
    input_size = 5

    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    train_x, train_y = build_dataset(train_sample)
    for epoch in range(epochs):
        model.train()
        watch_loss = []
        for i in range(train_sample // batch_size):
            x = train_x[i * batch_size:(i + 1) * batch_size]
            y = train_y[i * batch_size:(i + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("--------------\n第%d轮loss：%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), 'model.pt')

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
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s,  数据是第%d类" % (vec, 1+torch.argmax(res)))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.97889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
                [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.00797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.09349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]]
    predict("model.pt", test_vec)
