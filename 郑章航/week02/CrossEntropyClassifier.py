# coding:utf8
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

"""
任务描述
使用PyTorch框架实现一个简单的机器学习模型，以解决一个特定的分类问题。该问题要求根据输入向量的不同维度来确定其类别。
数据规则
输入数据：每个样本是一个5维的向量 x=[x1,x2,x3,x4,x5]。
输出标签：根据向量中最大值的位置来决定样本的类别。具体来说：
如果向量中的最大值出现在第一个位置（即 x1是最大的），则该样本属于类别0。
如果向量中的最大值出现在第二个位置（即 x2是最大的），则该样本属于类别1。
以此类推，直到第五个位置，共分为5个不同的类别。
当向量中有多个相同的最大值时，选择最靠前的最大值所对应的位置作为类别。
模型要求
损失函数：使用交叉熵损失函数。
输出层激活函数：使用Softmax函数，确保输出能够表示为概率分布。
增加一个隐藏层
任务目标：构建并训练一个神经网络模型，使其能够根据上述规则准确地对输入向量进行分类。
"""
# 设置环境变量以解决 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class TorchModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TorchModel, self).__init__()
        self.hidden = nn.Linear(input_size, 128)  # 增加一个隐藏层
        self.relu = nn.ReLU()  # 添加 ReLU 激活函数
        self.output = nn.Linear(128, output_size)  # 线性层
        self.softmax  = nn.Softmax(dim=1)  # Softmax激活函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.hidden(x)  # (batch_size, input_size) -> (batch_size, 128)
        x = self.relu(x)  # 应用 ReLU 激活函数
        x = self.output(x)  # (batch_size, 128) -> (batch_size, output_size)
        y_pred = self.softmax(x)  # (batch_size, output_size) -> (batch_size, output_size)
        if y is not None:
            return nn.CrossEntropyLoss()(y_pred, y.squeeze().long())  # 计算交叉熵损失
        else:
            return y_pred  # 输出预测结果

# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，按照最大值所在的位置，生成标签
def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)
    return x, max_index


# 随机生成一批样本
# 5个不同类别的样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.long)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    class_counts = [0] * 5  # 初始化每个类别的计数器
    correct, total = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        print(y_pred)
        y_pred_class = torch.argmax(y_pred, dim=1)
        correct = (y_pred_class == y).sum().item()
        total = y.size(0)
        for label in y:
            class_counts[label.item()] += 1
    accuracy = correct / total
    print("测试集中各类别样本数量：")
    for i, count in enumerate(class_counts):
        print("类别 %d: %d 个样本" % (i, count))
    print("正确预测个数：%d, 总样本数：%d, 正确率：%f" % (correct, total, accuracy))
    return accuracy


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    output_siz = 5  # 输出类别数
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size, output_siz)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "../model_epoch_20_cross_entropy.pth")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    output_size = 5
    model = TorchModel(input_size, output_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载训练好的权重
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        y_pred_class = torch.argmax(result, dim=1)  # 获取预测的最大概率位置
        for vec, res, prob in zip(input_vec, y_pred_class, result):
            # 使用 numpy.around 四舍五入概率值
            prob_rounded = np.around(prob.numpy(), decimals=6)
            print("输入：%s, 预测类别：%d, 概率分布：%s" % (vec, res, prob_rounded))
            print("概率分布和：", np.sum(prob.numpy()))  # 验证概率分布和是否为1


if __name__ == "__main__":
    main()
    test_vec = [
        [0.97889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
        [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
        [0.00797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
        [0.09349776, 0.59416669, 0.92579291, 0.41567412, 0.9958894],
        [0.12345678, 0.98765432, 0.23456789, 0.34567890, 0.45678901],
        [0.56789012, 0.67890123, 0.78901234, 0.89012345, 0.90123456]
    ]
    predict("../model_epoch_20_cross_entropy.pth", test_vec)
