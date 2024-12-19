import torch
import torch.nn as nn
import numpy as np
import random

class TorchModel(nn.Module):
    def __init__(self,input_size):
        super(TorchModel,self).__init__()
        self.linear=nn.Linear(input_size,5)     #输出5维向量
        self.loss = nn.CrossEntropyLoss()         #多分类交叉熵

    def forward(self,x,y=None):
        y_pred=self.linear(x)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred

#随机生成5维向量并获取最大值的索引
def build_sample():
    x=np.random.random(5)
    return x, np.argmax(x)

def build_dataset(sample_num):
    X=[]
    Y=[]
    for i in range(sample_num):
        x,y=build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model):
    model.eval()
    test_sample_num=100
    x,y=build_dataset(test_sample_num)

    with torch.no_grad():
        y_pred=model(x)
    print(y_pred)

def main():
    epoch_num=50
    batch_size=20
    train_sample=2000
    input_size=5
    learning_rate=0.01

    model = TorchModel(input_size)

    optim=torch.optim.Adam(model.parameters(),lr=learning_rate)
    log=[]
    train_x,train_y=build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss=[]
        for batch_index in range(train_sample//batch_size):
            x=train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print(res)  # 打印结果


if __name__ == "__main__":
    # main()

    test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.bin", test_vec)

    #输出结果
    #tensor([4.2123, -7.7212, -5.1284, -9.2869, 3.0490])
    #tensor([-5.4784, -8.8141, -2.8185, -3.0798, -4.4840])
    #tensor([-5.3931, 3.8447, -4.1474, -0.8775, -2.9529])
    #tensor([-7.2058, -0.4317, 3.9585, -3.2110, -6.9686])
