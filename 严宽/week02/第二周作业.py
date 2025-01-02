
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(TorchModel,self).__init__()
        self.layer1 = nn.Linear(input_size,hidden_size)
        self.layer2 = nn.Linear(hidden_size, 5)
        self.activation =torch.softmax   #激活函数 softmax
        self.loss = nn.CrossEntropyLoss()    #损失函数：交叉熵

    # 前向传播
    def forward(self,x,y = None):
        x = self.layer1(x)
        x = self.layer2(x)
        y_pred = self.activation(x,dim=1)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred


## 随机生成样本，样本中哪个数最大，返回哪个类别
def build_sample():
    x = np.random.random(5)
    maxX = max(x)
    if x[0] == maxX:
        return x,np.array([1,0,0,0,0])
    elif x[1] == maxX:
        return x,np.array([0,1,0,0,0])
    elif x[2] == maxX:
        return x,np.array([0,0,1,0,0])
    elif x[3] == maxX:
        return x,np.array([0,0,0,1,0])
    elif x[4] == maxX:
        return x,np.array([0,0,0,0,1])

## 随机生成样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x,y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X),torch.FloatTensor(Y)


def evaluate(model):
    model.eval()
    test_sample_num = 100
    x,y = build_dataset(test_sample_num)
    y1_num = 0
    y2_num = 0
    y3_num = 0
    y4_num = 0
    y5_num = 0
    for i in range(test_sample_num):
        if y[i].argmax() == 0:
            y1_num += 1
        elif y[i].argmax() == 1:
            y2_num += 1
        elif y[i].argmax() == 2:
            y3_num += 1
        elif y[i].argmax() == 3:
            y4_num += 1
        elif y[i].argmax() == 4:
            y5_num += 1
    print("本次预测集中共有%d样本，其中第一类样本%d个，第二类样本%d个，第三类样本%d个，第四类样本%d个，第五类样本%d个" % (test_sample_num, y1_num, y2_num, y3_num, y4_num, y5_num))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if y_p.argmax() == y_t.argmax():
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 20 #训练轮数
    batch_size = 20 #训练批次大小
    learning_rate = 0.2 #学习率
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    hidden_size1 = 3  # 隐藏层1维度
    model = TorchModel(input_size,hidden_size1)

    optim = torch.optim.SGD(model.parameters(), lr=learning_rate)
    log = []

    train_x,train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss= []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x,y)  # 计算损失函数
            loss.backward()   # 计算梯度
            optim.step()   # 更新权重
            optim.zero_grad()   #梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

def predict(model_path, input_vec):
    input_size = 5
    hidden_size = 3
    model = TorchModel(input_size,hidden_size)
    model.load_state_dict(torch.load(model_path,weights_only=True))
    print(model.state_dict())
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, res.argmax(), res.max()))  # 打印结果


if __name__ == "__main__":
    # main()
    test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.bin", test_vec)


