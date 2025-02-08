import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
改用交叉熵实现一个多分类任务，
五维随机向量最大的数字在哪一维，就属于哪一类
"""


class CrossEntropyWork(nn.Module):
    def __init__(self, input_size):
        super(CrossEntropyWork, self).__init__()
        self.linear = nn.Linear(input_size, 5)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


# 创建一个样本
def build_simple():
    x = np.random.random(5)
    max_index = np.argmax(x)
    return x, max_index


# 生成一个数据集
def build_dataset(total):
    X = []
    Y = []
    for i in range(total):
        x, y = build_simple()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))


# print(build_dataset(20))


def evaluate(model):
    model.eval()
    total_num = 100
    x, y = build_dataset(total_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_t, y_p in zip(y, y_pred):
            if y_t == torch.argmax(y_p):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d，准确率为%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    input_size = 5
    total_num = 5000
    learn_rate = 0.001
    epoch_num = 20
    batch_size = 20
    log = []
    model = CrossEntropyWork(input_size)

    optim = torch.optim.Adam(model.parameters(), lr=learn_rate)

    train_x, train_y = build_dataset(total_num)
    for e in range(epoch_num):
        model.train()
        watch_loss = []
        for i in range(total_num // batch_size):
            x = train_x[i * batch_size : (i + 1) * batch_size]
            y = train_y[i * batch_size : (i + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (e + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    torch.save(model.state_dict(), "CrossEntropyWorkModel.pt")
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    return


def predict(model_path, input_vec):
    model = CrossEntropyWork(input_size=5)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
        for vec, res in zip(input_vec, result):
            print("输入为：%s,预测类别：%s，概率值：%s" % (vec, torch.argmax(res), res))


if __name__ == "__main__":
    # main()
    test_vec = [
        [0.97889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
        [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
        [0.00797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
        [0.09349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894],
    ]
    predict("CrossEntropyWorkModel.pt", test_vec)
