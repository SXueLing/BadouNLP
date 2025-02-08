# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from pathlib import Path

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
需求：改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。

"""

# 继承nn.Module类，定义自己的模型
class TorchModel(nn.Module):
    # 初始化模型
    def __init__(self, input_size, output_size):
        # 调用父类初始化方法
        super(TorchModel, self).__init__()
        # 定义线性层，作用是将输入维度为input_size的向量映射到output_size维
        self.linear = nn.Linear(input_size, output_size)  # out_size上的概率分布
        # 定义激活函数，作用是将线性层输出的向量映射到0-1之间
        # 为什么要指定dim=1？
        # 输入向量维度是(batch_size, input_size)，输出向量维度是(batch_size, output_size)
        # 即(20,5) = 20行，每行5个元素
        # 因为交叉熵损失函数要求输入的预测值是概率分布，即每个元素之和为1
        # dim=1表示对每一行求和，使得每一行的元素之和为1
        # 如果指定dim=0，则对每一列求和，这样每一列的元素之和为1，而不是每一行的元素之和为1
        self.activation = nn.Softmax(dim=1)   # 使用cross_entropy损失函数，需要指定softmax，但实际上，softmax在交叉熵损失函数中已经包含了
        # nn.functional.cross_entropy 是一个函数，它直接计算输入张量和目标张量之间的交叉熵损失。
        # 这个函数通常用于前向传播时直接计算损失，不需要创建类实例。
        # nn.CrossEntropyLoss 是一个类，它继承自 nn.Module，并且需要创建一个实例。
        # 这个类不仅提供了计算交叉熵损失的功能，还允许你设置一些额外的参数，如权重、忽略的索引等。
        # 它更适合在定义模型时作为损失函数的一部分使用。

        # 下面这两种方式在这里的效果是一样的
        # 但是__init__方法中一般都是创建类实例，所以一般使用nn.CrossEntropyLoss
        # 如果要使用nn.functional.cross_entropy，则可以直接在forward方法中调用它
        # 而不需要在这里定义self.loss，因为nn.functional.cross_entropy是一个函数，不是类实例
        # self.loss = nn.functional.cross_entropy
        # self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    # forward方法 定义了神经网络的前向传播过程。
    # 具体来说，forward 方法描述了输入数据如何通过网络的各个层进行处理，最终生成输出。
    # 这个方法在每次调用模型时都会被自动调用，无论是训练阶段还是推理阶段。
    # 作用：
    #     1. 定义网络结构--线性层
    #     2. 计算损失
    #     3. 返回预测结果
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, input_size) -> (batch_size, output_size)
        y_pred = self.activation(y_pred)
        if y is not None:
            # return self.loss(y_pred, y)  # 预测值和真实值计算损失
            return nn.functional.cross_entropy(y_pred, y)  # 预测值和真实值计算损失
        else:
            # 这里也可以过一下激活层
            # return nn.functional.softmax(y_pred)  #  返回预测值, softmax函数将输出转换为概率分布,即每个元素在 (0, 1) 之间，并且所有元素之和为 1。
            return y_pred  # 输出预测结果,不经过softmax函数


def build_sample():
    x = np.random.random(5)
    # 获取最大值的索引
    # example：
    # >>> a = np.arange(6).reshape(2,3) + 10
    # >>> a
    # array([[10, 11, 12],
    #        [13, 14, 15]])
    # >>> np.argmax(a)   # 
    # 5
    # >>> np.argmax(a, axis=0)
    # array([1, 1, 1])
    # >>> np.argmax(a, axis=1)
    # array([2, 2])
    # np.argmax 函数用于返回数组中最大值的索引。
    # 不指定 axis 参数时，np.argmax 会将整个数组视为一个一维数组，并返回该一维数组中最大值的索引。
    # axis=None（默认）：将多维数组展平成一维数组，然后找到最大值的索引。
    # axis=0：沿着列方向查找每一列的最大值的索引。
    # axis=1：沿着行方向查找每一行的最大值的索引。
    max_index = np.argmax(x) # Returns the indices of the maximum values along an axis.
    # torch.argmax(x) 和np.argmax(x)区别：
    # torch.argmax(x) 返回的是张量中最大值的索引，返回的是一个整数。
    # np.argmax(x) 返回的是数组中最大值的索引，返回的是一个整数。
    # 如果要返回的是张量中最大值的索引，则可以使用torch.argmax(x)。
    # 如果要返回的是数组中最大值的索引，则可以使用np.argmax(x)。
    return x, max_index


def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        # mse_loss要求标签是浮点数
        # cross_entropy要求标签是整数
        # 所以这里的y没有加[]
        Y.append(y)  # 课堂上的loss是mse_loss，所以这里的y需要加[]
    # softmax和sigmoid的区别：
    # softmax：与输入具有相同的维度，每个元素在 (0, 1) 之间，并且所有元素之和为 1，适用于多分类任务。
    # sigmoid：与输入具有相同的维度，每个元素在 (0, 1) 之间，适用于二分类任务。
    return torch.FloatTensor(X), torch.LongTensor(Y)  # LongTensor 是一种整数类型的张量，用于存储标签数据。

# 评估模型
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():  # 禁用梯度计算
        y_pred = model(x)  # 进行模型预测获取到预测值
        # 预测值和真实值比较
        # y_pred.argmax() 返回张量中最大值的索引
        # y_pred.argmax() == y  比较预测值和真实值是否相等
        # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        # example：
        # >>> a = [1, 2, 3]
        # >>> b = [4, 5, 6]
        # >>> c = [7, 8, 9]
        # >>> zipped = zip(a, b, c)
        # >>> zipped
        # <zip object at 0x000001C8E4D8D940>
        # >>> list(zipped)
        # [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
        # >>> for i in zipped:
        # ...     print(i)
        # (1, 4, 7)
        # (2, 5, 8)
        # (3, 6, 9)  
        for y_pred,y_true in zip(y_pred,y):
            if y_pred.argmax() == y_true:  # argmax() 函数返回张量中最大值的索引。
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)   # 返回准确率


def main():
    epoch_num = 20           # 训练轮数
    batch_size = 20          # 每轮训练样本数
    train_sample = 5000      # 训练样本数
    input_size = 5           # 输入维度
    output_size = 5          # 输出维度
    learning_rate = 0.001    # 学习率
    model = TorchModel(input_size, output_size)    # 创建模型
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 选择优化器
    log = []
    train_x, train_y = build_dataset(train_sample)  # 构建训练集
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):  # 训练样本数除以每轮训练样本数，得到每轮训练的次数
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size] # 获取每轮训练的样本
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size] # 获取每轮训练的标签
            loss = model(x, y)  # 计算loss
            loss.backward()     # 反向传播，计算梯度
            optim.step()        # 更新权重
            optim.zero_grad()   # 梯度归零
            watch_loss.append(loss.item())    # 将loss值添加到watch_loss列表中
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))  # 打印每轮的平均loss
        acc = evaluate(model)    # 测试本轮模型的准确率
        log.append([acc, float(np.mean(watch_loss))])  # 将准确率和平均loss添加到log列表中
    # 保存模型
    # torch.save() 函数用于保存 PyTorch 模型的参数。
    # 该函数接受一个文件路径和一个包含模型参数的字典作为参数。
    # example：
    # >>> model = torch.nn.Linear(10, 2)
    # >>> torch.save(model.state_dict(), 'model.pth')
    # >>> loaded_model = torch.load('model.pth')
    # >>> model.load_state_dict(loaded_model)
    # Path(__file__).resolve().parent / "model.bin"：获取当前文件所在的目录，并拼接上 "model.bin" 文件名，得到保存模型的路径。
    # model.state_dict()：获取模型的所有参数，返回一个字典，其中键是参数的名称，值是参数的值。
    # example：
    # >>> model = torch.nn.Linear(10, 2)
    # >>> model.state_dict()
    # OrderedDict([('weight', tensor([[ 0.1234, -0.5678,  0.9101, -0.1123,  0.4567]])), ('bias', tensor([0.1234]))])
    torch.save(model.state_dict(), Path(__file__).resolve().parent / "model.bin")

    # plt.plot的用法
    # plt.plot(x, y, label="label")  # 绘制曲线
    # plt.xlabel("x轴标签")  # 设置x轴标签
    # plt.ylabel("y轴标签")  # 设置y轴标签
    # plt.title("标题")  # 设置标题
    # plt.savefig("filename.png")  # 保存图形
    # [l[0] for l in log]：将log列表中的每个元素的第一个值提取出来，形成一个新列表
    # [l[1] for l in log]：将log列表中的每个元素的第二个值提取出来，形成一个新列表
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 绘制准确率曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 绘制loss曲线
    plt.legend()  # 显示图例
    plt.show()    # 显示图形

def predict(model_path, input_vec):
    input_size = 5
    output_size = 5
    # 定义模型
    model = TorchModel(input_size, output_size)
    # 加载模型参数
    # torch.load() 函数用于加载 PyTorch 模型的参数。
    # 该函数接受一个文件路径作为参数，并返回一个包含模型参数的字典。
    # example：
    # >>> model = torch.nn.Linear(10, 2)
    # >>> torch.save(model.state_dict(), 'model.pth')
    # >>> loaded_model = torch.load('model.pth')
    # >>> model.load_state_dict(loaded_model)
    # >>> model.eval()
    # >>> input = torch.randn(1, 10)
    # >>> output = model(input)
    # >>> print(output)
    model.load_state_dict(torch.load(model_path))
    # model.eval() 函数用于将模型设置为评估模式。
    model.eval()
    with torch.no_grad():  # 禁用梯度计算
        # torch.FloatTensor(input_vec) 将输入向量转换为浮点数张量。
        # torch.no_grad() 函数用于禁用梯度计算，以节省内存和计算资源。
        # 在评估模型时，不需要计算梯度，因此可以使用 torch.no_grad() 函数来禁用梯度计算。
        # example：
        # >>> x = torch.randn(3, 4)
        # >>> with torch.no_grad():
        # >>>     y = x * 2
        # >>> print(y.requires_grad)
        # False
        # >>> print(x.requires_grad)
        # True
        # >>> print(y)
        # model.forward() 函数用于将输入数据传递给模型，并返回模型的输出。
        result = model.forward(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        # torch.argmax() 函数用于返回张量中最大值的索引，这里即预测类别。
        print("输入：%s, 预测类别：%d, 概率分布：%s" % (vec, res.argmax(), res))

if __name__ == "__main__":
    main()
    test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict(Path(__file__).resolve().parent / "model.bin", test_vec)
