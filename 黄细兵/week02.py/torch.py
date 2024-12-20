import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本

建造模型
损失函数

造测试数据

训练

测试
"""

# 构造模型
class TorchMoudle(nn.Module):
    def __init__(self, input_size):
        super(TorchMoudle, self).__init__()
        self.linear = nn.Linear(input_size, 5)    # 线性层
        self.activation = nn.Softmax()  # 激活函数
        self.loss = nn.functional.cross_entropy  # 损失函数

    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)
    return x, y

def build_dataset(total_num):
    X = []
    Y = []
    for i in range(total_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y)


def main():
    total_num = 5000  # 每轮多少个样本
    epoch_num = 20  # 训练次数
    batch_size = 20  # 批次
    lr = 0.01   # 学习率
    input_size = 5

    # 建立模型
    model = TorchMoudle(input_size)
    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # 测试数据
    train_x, train_y = build_dataset(total_num)

    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for i in range(total_num // batch_size):
            x = train_x[i * batch_size: (i + 1) * batch_size]
            y = train_y[i * batch_size: (i + 1) * batch_size]
            print(np.shape(x), np.shape(y))
            loss = model.forward(x, y)
            # 计算梯度
            loss.backward()
            # 权重更新
            optim.step()
            # 梯度归零
            optim.zero_grad()
            watch_loss.append(loss.item())

        print("============\n第%d伦，平均loss为%f" % (epoch+1, np.mean(watch_loss)))
        # 测试本轮模型效果
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), 'model.pt')
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


def evaluate(model):
    model.eval()  # 测试模式
    eval_num = 100
    x, y = build_dataset(eval_num)
    #print("本次测试，正确样本为：%d个，错误样本为：%d" % (sum(y), eval_num-sum(y)))

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model.forward(x)
        for y_p, y_t in zip(y_pred, y):
            if np.argmax(y_p) != y_t:
                wrong += 1
            else:
                correct += 1
    print("正确预测个数：%d，正确率：%f" % (correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def test(path, data):
    input_size = 5
    model = TorchMoudle(input_size)
    model.load_state_dict(torch.load(path))
    print("加载已训练模型=================")
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        y_pred = model.forward(torch.FloatTensor(data))
        for y_p, y_t in zip(y_pred, data):
            print("输入：%s，类型：%d" % (y_t, np.argmax(y_p)))



if __name__ == '__main__':
   main()
   #  test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
   #              [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
   #              [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
   #              [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
   # test("model.pt", test_vec)
