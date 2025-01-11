# coding:utf8
import torch
import torch.nn as nn
import random
import numpy as np
import json
import matplotlib.pyplot as plt

'''
    输入一个5维向量，其中第几个数最大即结果预测为第几类
'''
class PreType(nn.Module):
    def __init__(self,input_size):
        super(PreType,self).__init__()
        # 线性层
        self.linear = nn.Linear(input_size,5)
        # 激活函数
        # self.activation = torch.softmax
        # loss函数 cross_entropy 内包含 softmax
        self.loss = nn.CrossEntropyLoss()

    def forward(self,x,y_true=None):
        y_pre=self.linear(x)

        if y_true is not None:
            return self.loss(y_pre,y_true)
        else:
            return y_pre

# 构造样例数据
def get_sample():
    x=np.random.random(5)
    return x,np.argmax(x)
# 构造数据集

def get_dataset(nums):
    X = []
    Y = []
    for i in range(nums):
        x,y = get_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X),torch.LongTensor(Y)

# 测准确率
def evaluate(model):
    model.eval()
    test_nums = 512
    x,y_true = get_dataset(test_nums)
    correct,wrong = 0,0
    with torch.no_grad():
        y_pre = model(x)
        for y_pre,y_true in zip(y_pre,y_true):
            if torch.argmax(y_pre) == int(y_true):
                correct += 1
            else:
                wrong += 1
    print('预测正确个数为: %d,错误个数为:%d,正确率为： %f' %(correct,wrong,correct/(correct+wrong)))
    return correct/(correct+wrong)
# 模型开始训练
def main():
    epoch_num = 10
    batch_size = 50
    nums    = 50000
    learning_rate = 0.1
    input_size = 5
    # 实例化一个模型
    model = PreType(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    #构造数据集
    train_x,train_y = get_dataset(nums)
    for epoch in range(epoch_num):
        model.train()
        each_loss = []
        for i in range(nums//batch_size):
            x = train_x[ i * batch_size : (i + 1) * batch_size]
            y = train_y[ i * batch_size : (i + 1) * batch_size]
            loss = model.forward(x,y) #计算 loss
            loss.backward()  #计算梯度
            optim.step()     #计算权重
            optim.zero_grad() #梯度归0
            each_loss.append(loss.item())
        print( "********第%d轮，当前loss为%f********" % (epoch + 1 , np.mean(each_loss)))
        evaluate(model)
    torch.save(model.state_dict(),'week2.model')
    return

def predict(model_path,input_vec):
    input_size = 5
    model = PreType(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        # result = softmax(result)
    for vec, res in zip(input_vec, result):
        # print(type(torch.argmax(res)))
        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, torch.argmax(res), res))  # 打印结果

if __name__ == '__main__':
    main()
    test_vec = [[0.97889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
                [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.20797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.29349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]]
    predict("week2.model", test_vec)
