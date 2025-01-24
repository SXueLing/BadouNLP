'''
Author: Zhao
Date: 2024-12-18 18:21:57
LastEditTime: 2024-12-18 20:15:12
FilePath: myTorchDemo1.py
Description: 

'''
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
"""
Task:
基于pytorch框架编写模型训练
规律：x是一个5维向量，获取最大的数字所在的那一维
"""

class MyTorchModle(nn.Module):
    def __init__(self,input_size):
        super(MyTorchModle, self).__init__()
        self.linear = nn.Linear(input_size, 5)
        self.activation = nn.Softmax(1) #归一化函数
        self.lose = nn.functional.cross_entropy #loss函数用交叉熵计算
        
    # 输入真实值返回loss，否则返回预测值
    def forward(self,x,y=None):
        y_pre = self.linear(x)
        if y is not None:
            return self.lose(y_pre, y)
        else:
            return self.activation(y_pre)
    

#构建随机生成样本 输出最大的索引
def build_sample():
    x = np.random.random(5)
    index = np.argmax(x)
    return x,index

#批量构建样本
def build_dateset(train_sample):
    X = []
    Y = []
    for i in range(train_sample):
        x,y = build_sample()
        X.append(x)
        Y.append(y)
        X_array = np.array(X)
        Y_array = np.array(Y)
    return torch.FloatTensor(X_array),torch.LongTensor(Y_array)


# 用来测试每轮模型的准确率  
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dateset(test_sample_num)
    print("本轮预测集中共有%d个正样本，%d个负样本" % (sum(y),test_sample_num-sum(y)))
    correct,wrong = 0, 0
    with torch.no_grad():
        y_pre = model(x)
        for y_p,y_true in zip (y_pre,y):
            # torch.argmax是求最大数所在维
            if torch.argmax(y_p) == int(y_true):
                correct += 1
            else:
                wrong += 1
    print("正确预测的个数为: %d，正确率为: %f" % (correct, correct/ (correct + wrong)))
    return correct / (correct + wrong)


def main():
    epoch_num = 20
    #每次训练样本数
    batch_size = 20
    #每轮总样本数
    train_sample = 5000
    #输入向量维度
    input_size = 5
    #学习率
    learning_rate = 0.001
    #建立模型
    model = MyTorchModle(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(),learning_rate)
    log = []
    train_x, train_y = build_dateset(train_sample)
    # 训练过程
    for i in range(epoch_num):
        model.train
        watch_loss = []
        for index in range(train_sample // batch_size):
            x = train_x[index * batch_size : (index + 1) * batch_size]
            y = train_y[index * batch_size : (index + 1) * batch_size]
            # 计算loss
            loss = model(x,y)
            # 计算梯度
            loss.backward()
            # 更新权重
            optim.step()
            # 梯度归零
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("========>\n 第%d轮平均loss为：%f" %(i+1,np.mean(watch_loss)))
        # 测试本轮模型结果
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(),"model.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [i[0] for i in log], label="line acc")
    plt.plot(range(len(log)), [i[1] for i in log], label = "line lose")
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def my_predict(model_path, input_vec):
    input_size = 5
    model = MyTorchModle(input_size)
    # 加载训练好的权重
    model.load_state_dict(torch.load(model_path, weights_only=True))
    #print(model.state_dict())
    # 测试模式
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    for vec, res in zip (input_vec, result):
        print("输入: %s,  预测类别: %s, 概率: %s" % (vec, torch.argmax(res), res))

if __name__ == "__main__":
    #main()
    test_vec= [[0.82758743, 0.94952515, 0.8724857,  0.24265896, 0.66460911],
                [0.56211635, 0.21982102, 0.85189468, 0.18419216, 0.07659104],
                [0.48276522, 0.81470027, 0.16664831, 0.33291732, 0.78996745],
                [0.37547093, 0.89525947, 0.11272405, 0.83698723, 0.48894183]]
    
    """打印结果：
    输入: [0.82758743, 0.94952515, 0.8724857, 0.24265896, 0.66460911],  预测类别: tensor(1), 概率: tensor([0.2466, 0.3763, 0.2329, 0.0272, 0.1170])
    输入: [0.56211635, 0.21982102, 0.85189468, 0.18419216, 0.07659104],  预测类别: tensor(2), 概率: tensor([0.2168, 0.0545, 0.6399, 0.0547, 0.0342])
    输入: [0.48276522, 0.81470027, 0.16664831, 0.33291732, 0.78996745],  预测类别: tensor(1), 概率: tensor([0.1193, 0.4276, 0.0322, 0.0717, 0.3492])
    输入: [0.37547093, 0.89525947, 0.11272405, 0.83698723, 0.48894183],  预测类别: tensor(1), 概率: tensor([0.0584, 0.4607, 0.0228, 0.3539, 0.1042])"""

    my_predict("model.pt",test_vec)