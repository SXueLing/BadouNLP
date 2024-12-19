from AI.week2.week2_homework.data_builder import dataset_builder
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
"""
使用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。
"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵
    def forward(self, x,y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果

def main(learning_rate=0.25):
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 50  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数,
    input_size = 5  # 输入向量维度
    learning_rate = learning_rate  # 学习率，学习率对预测准确性有较大影响
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = dataset_builder(train_sample)
    # 训练过程
    for epoch in range(epoch_num):  # 一共要进行 epoch_num 轮
        model.train()
        watch_loss = [] # 每轮都初始化一遍loss
        for batch_index in range(train_sample // batch_size):    #  每轮将所有样本按批次训练，每批次更新一次权重
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            #神经网络训练核心
            loss = model.forward(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            # 神经网络训练核心
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))    #对每轮的 loss 求平均值
    return model

def test(model):
    """
    整体思路是，将一组测试数据 x 放入模型，得到 y_pred
    然后将 y_pred 中正确的数量 / y_true 总数量，得到准确率
    """
    model.eval()    #进入评估状态
    x, y_true = dataset_builder(100)    # 准备测试数据。y_true是一个LongTensor类型
    y_pred = model(x)   # 得到预计的 y_pred。这时 y 是一个包含五维向量的数组
    _, y_pred = torch.max(y_pred.data, 1)   #这时主要是将 y_pred从一个五维向量数组修改一个表达方式，变成 LongTensor 的表达方式。便于比较。
    total = y_true.size(0)  # 获得分母
    correct = (y_pred == y_true).sum().item()   # 获得分子
    c_r =  correct / total * 100
    print('模型预测的正确率为：{}'.format(c_r))
    return  c_r


def find_better_lr():
    """
    这个函数主要通过循环的方式，将学习率从 0.1到 0.5，按照 0.01的步长逐步训练模型，探索对准确率比较高的学习率
    函数最后生成图像，x为学习率，y为准确性
    """
    x = []  # 存放学习率
    y = []  # 存放准确性
    for i in range(10,51):  #学习率的变化范围为 0.1 -0.5
        lr = i/100  # 转化为学习率
        x.append(lr)    # 存放学习率
        y.append(test(main(lr)))    # 存放准确性
    plt.plot(x, y)  # 绘图
    plt.show()  #展示图片

if __name__ == '__main__':
    find_better_lr()
