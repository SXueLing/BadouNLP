# coding:utf-8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import accuracy_score

'''

改用交叉熵实现一个多分类任务，五维随机向量最大的数在哪维就属于哪一类

'''
class Net_classify(nn.Module):
    def __init__(self,input_size,dense):
        super(Net_classify,self).__init__()
        self.fc1 = nn.Linear(input_size,dense) # 第一个隐藏层
        self.fc2 = nn.Linear(dense,5) # 第二个隐藏层
        self.relu = nn.ReLU() # 激活函数

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

def generate_data():
    x = np.random.rand(1000,5)
    y = np.argmax(x,axis=1)
    y = y.astype(int)
    return x,y

def train():
    x,y = generate_data()
    x = torch.tensor(x,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.long)
    model = Net_classify(x.shape[1],10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    batch_size = 100
    losses = []
    accuracies = []
    for epoch in range(100):
        loss = 0
        for i in range(x.shape[0]//batch_size):
            optimizer.zero_grad()
            outputs = model(x[i*batch_size:(i+1)*batch_size,:])
            loss = criterion(outputs,y[i*batch_size:(i+1)*batch_size])
            loss.backward()
            optimizer.step()
            loss+=loss.item()
        losses.append(loss.detach().numpy())
        print('epoch:',epoch,'loss:',loss)

        # 测试模型
        model.eval()
        with torch.no_grad():
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            accuracy = accuracy_score(y.numpy(), predicted.numpy())
            accuracies.append(accuracy)
            print(f'Accuracy: {accuracy:.4f}')

    return losses,accuracies


if __name__ == '__main__':
    losses,accuracies = train()
    plt.subplot(211)
    plt.plot(losses,label='loss')
    plt.legend()
    plt.subplot(212)
    plt.plot(accuracies,label='accuracy')
    plt.legend()
    plt.show()
