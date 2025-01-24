import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


class TorchModel(nn.Module):
    def __init__(self, input_size=5, hidden_size1=16, hidden_size2=5):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.activation1 = torch.sigmoid  # 激活函数 sigmoid relu, tanh
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.activation2 = torch.sigmoid  # 激活函数 sigmoid relu, tanh
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        y_pred = x
        if y is not None:
            return self.loss(y_pred, y)
        return y_pred


def build_dataset(total_sample_num: int):
    X = torch.randn(total_sample_num, 5)
    m = torch.argmax(X, dim=1)
    Y = torch.zeros(X.shape)
    for i, j in enumerate(m):
        Y[i][j] = 1
    return X, Y


def evaluate(model, test_sample_num=100):
    model.eval()
    X, Y = build_dataset(test_sample_num)
    with torch.no_grad():
        y_pred = model(X)
    total_true = (torch.argmax(Y, dim=1) == torch.argmax(y_pred, dim=1)).sum()
    return total_true / test_sample_num


def train(model, epoch_num=100, batch_size=32, lr=0.001, train_sample_num=5000):
    optim = torch.optim.Adam(model.parameters(), lr)
    log = []
    train_x, train_y = build_dataset(train_sample_num)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample_num // batch_size):
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()  # 反向传播求梯度
            optim.step()  # 更新参数
            optim.zero_grad()  # 清空梯度
            watch_loss.append(loss.item())
        acc = evaluate(model)
        print(f"========= 第{epoch + 1}轮平均loss:{np.mean(watch_loss)}, {acc}, {watch_loss}")
        log.append([acc, float(np.mean(watch_loss))])
    torch.save(model.state_dict(), "model-multi-classification.bin")
    print(f"训练结束: {model.state_dict()}")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    return


def predict(model_path, input_vec):
    model = TorchModel(5, 16, 5)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    with torch.no_grad():
        model.eval()
        result = model.forward(torch.FloatTensor(input_vec))
        return result


if __name__ == '__main__':
    model = TorchModel(5, 16, 5)
    epoch_num = 100
    batch_size = 32
    lr = 0.001
    train_sample_num = 5000
    train(model, epoch_num, batch_size, lr, train_sample_num)

    # X, Y = build_dataset(10)
    # print(X)
    # print(Y)
    # print(Y.argmax(dim=1))

    test_vec = [[0.7916, -0.3871, 0.7240, 1.0983, 1.1978],
                [-1.1679, 0.0590, -0.7157, -0.4546, 0.4651],
                [1.0421, -1.4812, 0.3756, -0.8054, 0.3562],
                [-1.6474, -0.1069, 1.5764, -0.7526, -0.7142],
                [0.0106, 0.6019, -0.3546, 1.1071, -0.7238],
                [0.1655, -0.0667, 0.2304, 0.7414, 0.8280],
                [-0.3704, 1.0814, -1.3030, 1.1144, 0.3818],
                [1.3124, -0.8597, 0.2891, 1.9811, 0.2592],
                [-0.7410, -0.0234, 1.3114, -1.3043, -0.4490],
                [-1.5006, -0.6717, -0.1640, -0.7064, 0.1387]]
    y_true = [4, 4, 0, 2, 3, 4, 3, 3, 2, 4]
    result = predict("model-multi-classification.bin", test_vec)
    for r in result:
        print(r.max(dim=0).indices, r, r.sum())
    pass
