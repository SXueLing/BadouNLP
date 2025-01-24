# coding:utf8

import numpy as np
import torch
import torch.nn as nn

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，5维向量最大的数字在哪维就属于哪一类

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)
        y_pred = self.activation(x)  # (batch_size, 5) -> (batch_size, 5)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，5维向量最大的数字在哪维就属于哪一类
def build_sample():
    x = np.random.random(5)
    x_max = max(x)
    if x_max == x[0]:
        return x, 0
    elif x_max == x[1]:
        return x, 1
    elif x_max == x[2]:
        return x, 2
    elif x_max == x[3]:
        return x, 3
    elif x_max == x[4]:
        return x, 4


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    unique_elements, counts = torch.unique(y, return_counts=True)
    print("本次预测集中：")
    for value, count in zip(unique_elements, counts):
        print("第%d类出现了%d次" % (value.item(), count.item()))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for true_label, pred_label in zip(y, y_pred):
            pred_label = torch.argmax(pred_label)
            if int(true_label) == int(pred_label):
                correct += 1
            else:
                wrong += 1
    print("正确预测样本数：%d, 正确率：%.5f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
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
        print("====================================\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    # print(log)
    # plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    # plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    # plt.legend()
    # plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, torch.argmax(res), res[torch.argmax(res)]))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.04687492, 0.27014328, 0.37713261, 0.67787083, 0.50957651],
                [0.82121108, 0.21276414, 0.67325434, 0.52817481, 0.12674698],
                [0.57312255, 0.60227138, 0.69300448, 0.68951876, 0.92448493],
                [0.84070541, 0.02046641, 0.51502633, 0.37787752, 0.42062591],
                [0.36916909, 0.80970502, 0.87877214, 0.52106434, 0.54852027]]
    predict("model.bin", test_vec)
