# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：一个五维向量 哪一个数字大 就输出第几类

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)
        if y is not None:
            return self.loss(x, y)  # 预测值和真实值计算损失
        else:
            return x  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，生成这个向量是哪一个类别  返回参数暂时用[1,0,0,0,0]
def build_sample():
    x = np.random.random(5)
    y = x.copy()
    y.sort()
    compare = y[-1]
    for index, value in enumerate(x):
        if value == compare:
            return x, index


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


groupDict = {}
groupDict["第一类"] = 0
groupDict["第二类"] = 1
groupDict["第三类"] = 2
groupDict["第四类"] = 3
groupDict["第五类"] = 4


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 1000
    x, y = build_dataset(test_sample_num)
    groups = {}
    for vector in y:
        key = int(vector)  # 将张量转为可哈希的元组作为键
        if key not in groups:
            groups[key] = 0
        groups[key] += 1
    # 输出分组结果
    for keyDic, valueDic in groupDict.items():
        for key, vectors in groups.items():
            if (valueDic == key):
                print(f"类别 {keyDic}，数量 {vectors}")

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if check_index(y_p, y_t):
                correct += 1  # 负样本判断正确
            else:
                wrong += 1
    percent = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def check_index(p1: list, index_p2: int):
    max_value_p1 = max(p1)
    index_p1 = 0
    for index, value in enumerate(p1):
        if value == max_value_p1:
            index_p1 = index
    return index_p1 == index_p2


def main():
    # 配置参数
    epoch_num = 60  # 训练轮数
    batch_size = 40  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
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
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "second_week_train.bin")
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
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        model_result_type = get_tag(res)
        input_type = get_tag(vec)
        print("输入：%s, 预测类别：%d ,是否准确 %s" % (vec, model_result_type, model_result_type == input_type))  # 打印结果


def get_tag(p: list):
    max_value_p1 = max(p)
    index_p2 = 0
    for index, value in enumerate(p):
        if value == max_value_p1:
            index_p2 = index
    return index_p2


if __name__ == "__main__":
    main()
    # X, Y = build_dataset(100)
    # predict("second_week_train.bin", X)
    pass
