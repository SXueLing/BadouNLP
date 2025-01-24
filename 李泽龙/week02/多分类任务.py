# coding:utf8
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数是最大的数，则为第一类样本；第2个数是最大的数，则为第二类样本；以此类推分为五大类

"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  #线性层
        self.loss = nn.CrossEntropyLoss()  #计算损失
        
    def forward(self, x, y = None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)   #计算损失
        else:
            return y_pred   #输出预测结果
        

#单个样本内容
def data_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)
    return x, max_index
    
#生成样本个数，初始化样本数据
def build_dataset(datasetsample_num):
    X = []
    Y = []
    for i in range(datasetsample_num):
        x,y = data_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X),torch.LongTensor(Y)

#测试代码
def evaluate(model):
    model.eval()
    testsample_num = 100
    x,y = build_dataset(testsample_num)
    counts = torch.bincount(y, minlength=5)  # 我们指定minlength=5，因为类别是从0到4
    print("本次测试样本中第一类样本为:%d\t第二类样本为:%d\t第三类样本为:%d\t第四类样本为:%d\t第五类样本为:%d\t" % (counts[0],counts[1],counts[2],counts[3],counts[4]))
    correct, error = 0 , 0
    with torch.no_grad():
        y_pred = model(x) #模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):
            if np.argmax(y_p) == y_t:
                correct += 1
            else:
                error += 1
    print("正确预测个数： %d, 正确率为： %f" % (correct, correct / (correct + error)))
    return correct / (correct + error)
        
    
#训练模型
def main():
    #定义超参数
    sample_num = 5000  #训练样本个数
    lr_rate = 0.001  #学习率
    epoch_num = 30 #训练轮数
    batch_size = 20  #批次大小
    input_size = 5  #输入大小
    #确定模型
    model = TorchModel(input_size)
    #模型训练
    #定义优化器
    optim = torch.optim.Adam(model.parameters(), lr= lr_rate)
    log = []
    #创建训练集
    x_train, y_train = build_dataset(sample_num)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(sample_num // batch_size):
            x = x_train[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = y_train[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  #计算loss  model.forward(x,y)
            loss.backward()  #计算梯度
            optim.step()  #更新权重
            optim.zero_grad()  #梯度归零
            watch_loss.append(loss.item())
        print("------\n第%d轮平均loss为：%f" % (epoch+1, np.mean(watch_loss)))
        acc = evaluate(model) 
        log.append([acc, float(np.mean(watch_loss))])
        
    #保存训练模型
    #torch.save(model.state_dict(), "model1.bin")    
    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label = 'acc')
    plt.plot(range(len(log)), [l[1] for l in log], label = "loss")
    plt.legend()
    plt.show()
    return
     
#利用保存好的训练模型预测结果     
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d" % (vec, np.argmax(res)))  # 打印结果

if __name__ == "__main__":
    #main()
    test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.84890681,0.95520434,0.98758807],
                [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model1.bin", test_vec)
    
         
