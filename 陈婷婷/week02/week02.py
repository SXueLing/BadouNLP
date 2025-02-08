'''
规律：x是一个5维向量，最大的值索引位于第几位便属于哪一类
'''

import torch
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt

#定义模型
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        #全连接层、softmax激活函数、交叉熵损失函数
        #self.linear = nn.Linear(input_size, 5)
        self.layer = nn.Linear(input_size, 5) 
        self.activate = torch.relu
        self.loss = nn.functional.cross_entropy
    
    #前向传播
    def forward(self, x, y=None):
        x = self.layer(x) #shape: (batch_size, input_size) -> (batch_size, hidden_size1) 
        y_pred = self.activate(x) 
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

#定义样本
def data_sample():
    x = np.random.random(5)
    idx = np.argmax(x)
    y = np.zeros(5)
    y[idx] = 1
    return x, y

def build_data(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = data_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)

#模型测试
def model_test(model):
    model.eval()
    test_sample_num = 200
    x, y = build_data(test_sample_num)
    t, f = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if y_t.nonzero() == y_p.argmax():
                t += 1
            else:
                f +=1
    print("正确预测个数：%d, 正确率：%f" % (t, t / (t + f)))
    return t / (t + f)

#模型预测
def model_predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path)) #加载训练好的模型权重
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        y_p = res.argmax() + 1
        y_t = np.argmax(vec) + 1
        print("输入：%s, 实际类别：%d, 预测类别：%d" % (vec, y_t, y_p))  # 打印结果

def main():
    #参数配置
    batch_size = 100
    epoch_num = 50
    train_sample = 50000
    input_size = 5
    learning_rate = 0.01

    #建立模型
    model = TorchModel(input_size)
    #优化器
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    #模型训练
    train_x, train_y = build_data(train_sample)
    for epoch in range(epoch_num):
        model.train()
        epoch_loss = []
        for idx in range(train_sample//batch_size):
            x = train_x[batch_size*idx : batch_size*(idx+1)]
            y = train_y[batch_size*idx : batch_size*(idx+1)]
            loss = model(x, y) 
            loss.backward() #计算梯度
            opt.step() #更新权重
            opt.zero_grad() #权重归零
            epoch_loss.append(loss.item())
        print("------------------第%d轮平均loss:%f", epoch+1, np.mean(epoch_loss))
        acc = model_test(model)
        log.append([acc, np.mean(epoch_loss)])
    #模型保存
    torch.save(model.state_dict(), "model.bin")

    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

if __name__ == "__main__":
    main()
    test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
                 [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                 [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                 [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    model_predict("model.bin", test_vec)
