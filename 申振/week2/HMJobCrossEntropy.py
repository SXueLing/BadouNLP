import numpy as np
import torch

import torch.nn as nn
from matplotlib import pyplot as plt


# 使用交叉熵实现一个多分类任务

class TorchModel(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.linear = nn.Linear(input_size,10)
        self.loss = nn.CrossEntropyLoss()
    def forward(self, x,y=None):
        x = self.linear(x)
        if y is not None:
            loss = self.loss(x,y)
            return loss
        return x

def build_sample():
    x = np.random.rand(5)
    max_index = np.argmax(x)
    return x, max_index

def build_dataset(sample_num):
    X,Y = [],[]
    for i in range(sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct,wrong = 0,0
    with torch.no_grad():
        y_pred = model(x)
        for y_p,y_t in zip(y_pred,y):
            y_p = np.argmax(y_p)
            if int(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("本次正确预测个数：%d,正确率：%f" % (correct,(correct/(correct+wrong))))
    return correct/(correct+wrong)

# 使用训练好的模型做预测
def predict(model_path,input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    for vec,res in zip(input_vec,result):
        res = np.argmax(res)
        print("输入：%s，预测类别：%d" % (vec,res))


def main():
    # 配置参数
    epoch_num = 1000
    sample_num = 1000
    batch_size = 10
    lr = 1e-3
    input_size = 5

    # 初始化模型
    model = TorchModel(input_size)
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log = []
    # 生成训练样本
    train_x, train_y = build_dataset(sample_num)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(sample_num//batch_size):
            x = train_x[batch_index*batch_size:(batch_index+1)*batch_size]
            y = train_y[batch_index*batch_size:(batch_index+1)*batch_size]
            loss = model.forward(x,y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        print("第%d轮平均loss：%f" % (epoch+1, np.mean(watch_loss)))
        acc = evaluate(model)
        #记录结果
        log.append([acc,np.mean(watch_loss)])
    torch.save(model.state_dict(), "./model.pth")
    plt.plot(range(len(log)), [l[0] for l in log],label="acc")
    plt.plot(range(len(log)), [l[1] for l in log],label="loss")
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    # main()

    test_vec = [[0.97889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
                [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.00797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.09349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]]
    predict("./model.pth", test_vec)
