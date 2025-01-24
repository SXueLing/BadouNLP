import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class torchModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(torchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.Sigmoid() # 等于torch.sigmoid
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y = None):
        x = self.linear1(x)
        x = self.activation(x)
        y_pred = self.linear2(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_samples():
    x = np.random.random(5)
    max_num = x[0]
    idx = 0
    for i in range(1, len(x)):
        if x[i] > max_num:
            idx = i
            max_num = x[i]
    return x, idx

def build_dataset(total_nums):
    X = []
    Y = []
    for i in range(total_nums):
        x, idx = build_samples()
        X.append(x)
        Y.append(idx)
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model):
    model.eval()
    test_num = 500
    test_x, test_y = build_dataset(test_num)
    correct = 0
    y_pred = model.forward(test_x)
    for y_p, y_t in zip(y_pred, test_y):
        pre_label = torch.argmax(y_p)  # 返回最大值中的第一个最大值
        if pre_label == y_t:
            correct += 1
    print("测试样本个数为%d, 正确预测个数: %d, 正确率: %f" % (test_num, correct, correct / test_num))
    return correct / test_num


def main():
    epoch_nums = 100
    total_nums = 5000
    lr = 0.1
    batch_size = 64
    inputSize = 5
    hiddenSize = 128
    outputSize = 5
    log = []

    model = torchModel(inputSize, hiddenSize, outputSize)
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    train_x, train_y = build_dataset(total_nums)

    for epoch in range(epoch_nums):
        model.train()
        watch_loss = []
        for batch_index in range(total_nums // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model.forward(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item()) # loss.item()将张量转化为标量
        print("--------\n第{}轮Epoch平均loss: {}".format(epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, np.mean(watch_loss)])

    torch.save(model.state_dict(), "model.pt")
    # 画图

    plt.plot(range(len(log)), [l[0] for l in log], label = "accuracy")
    plt.plot(range(len(log)), [l[1] for l in log], label = "loss")
    plt.legend()
    plt.show()

def predict(model_path, test_vec):
    input_size = 5
    hidden_size = 128
    output_size = 5
    model = torchModel(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        y_pred = model.forward(torch.Tensor(test_vec))
    for vec, y_p in zip(test_vec, y_pred):
        y_t = vec.index(max(vec))
        pre_label = torch.argmax(y_p).item()
        print("输入: %s, 预测类别: %d, 实际类别: %d" % (vec, pre_label, y_t))


if __name__ == "__main__":
    main()

    test_vec = [[0.97889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
                [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.00797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.09349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]]
    predict("model.pt", test_vec)
