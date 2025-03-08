import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self,input_size):
        super(TorchModel,self).__init__()
        self.linear = nn.Linear(input_size,5)
        self.loss_fun = nn.functional.cross_entropy
    
    def forward(self,x,y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss_fun(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果

def build_sample():
    x = np.random.random(5)
    y = np.zeros(5,dtype=int)
    y_index = np.argmax(x)
    y[y_index] = 1
    return x,y

def Build_data(test_num):
    X = []
    Y = []
    for i in range(test_num):
        x,y = build_sample()
        X.append(x)
        Y.append(y)
    return  torch.FloatTensor(X), torch.FloatTensor(Y)

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = Build_data(test_sample_num)
    print("本次预测集中共有10000个样本")
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        _, y_max = torch.max(y, dim=1)
        _,y_pred_max = torch.max(y_pred, dim=1)
        for y_p, y_t in zip(y_pred_max, y_max):  # 与真实标签进行对比
            if y_p == y_t :
                correct += 1  # 负样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return  correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 100
    batch_size = 20
    train_num = 5000
    input_size = 5
    learn_rate = 0.001

    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learn_rate)
    log = []
    train_x,train_y = Build_data(train_num)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for i in range(train_num // batch_size):
            x = train_x[i * batch_size : (i+1) * batch_size]
            y = train_y[i * batch_size : (i+1) * batch_size]
            loss = model.forward(x,y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return 1
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for ves,res in zip(input_vec,result):
        print("输入：%s 预测类别：%s, 概率值：%s" % (ves, np.argmax(res) , torch.round(torch.nn.functional.softmax(res)*100)/100)) # 打印结果
if __name__ == "__main__":
    main()
    # test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.bin", test_vec)

