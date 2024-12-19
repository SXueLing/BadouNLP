# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果5个数中最大的为正样本，反之为负样本，如果一样大，前面的数为正样本，其余为负样本

"""
class MultiClassficationModel(nn.Module):
    def __init__(self,input_size):
        super(MultiClassficationModel, self).__init__()
        self.linear = nn.Linear(input_size,5) # 线性层
        self.loss = nn.functional.cross_entropy #损失函数

        # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果
def  build_simple():
    x = np.random.random(5)
    # 获取最大值的索引
    max_index = np.argmax(x)
    return x, max_index
# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_simple()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


#测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_samlpe_num = 100
    x,y = build_dataset(test_samlpe_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p,y_t in zip(y_pred,y):
            if torch.argmax(y_p) == int(y_t):
                correct +=1
            else:
                wrong +=1
    print("预测正确个数：%d.正确率：%f" % (correct,correct / (correct + wrong)))
def main():
    # 配置参数
    epoch_num = 20 #训练论数
    batch_size = 20 #每次训练样本数
    train_sample = 5000 #每轮训练的样本总数
    input_size = 5 # 输入向量维度
    learning_rate = 0.001
    # 建立模型
    model = MultiClassficationModel(input_size) # 多分类模型
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 读取训练集
    train_x,train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index*batch_size:(batch_index+1)*batch_size]
            y = train_y[batch_index*batch_size:(batch_index+1)*batch_size]
            loss = model(x,y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step() # 更新权重
            optim.zero_grad() #梯度归零
            watch_loss.append(loss.item())
        print("=======\n第%d轮平均loss%f" % (epoch +1,np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc,float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(),"model.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)),[l[0] for l in log],label="acc")
    plt.plot(range(len(log)),[l[1] for l in log],label="loss")
    plt.legend()
    plt.show()
    return
def predict(model_path,input_vec):
    input_size = 5
    model = MultiClassficationModel(input_size)
    model.load_state_dict(torch.load(model_path)) #加载训练好的权重
    print(model.state_dict())
    model.eval() #测试模型
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec,result):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, torch.argmax(res), res))

if __name__ == '__main__':
    main()
    test_vec = [[0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.4963533, 0.5524256, 0.95758807, 0.65520434, 0.84890681],
                [0.48797868, 0.67482528, 0.13625847, 0.34675372, 0.09871392],
                [0.49349776, 0.59416669, 0.92579291, 0.41567412, 0.7358894]]
    predict("model.pt", test_vec)


