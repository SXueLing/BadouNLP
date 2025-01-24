# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
任务：5分类任务，共有5个类别
规律：x是一个5维向量，最大数所在索引为对应类别：0,1,2,3,4
在描述中我们将其分别记为a,b,c,d,e类

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        # self.activation = torch.softmax(dim=1) # softmax激活函数
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        # y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(x, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(x, dim=1)  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，最大值对应的索引为相应类, 若有多个最大值, 设定其中最小索引为相应类
def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)
    y = np.zeros(5)
    y[max_index] = 1
    return x, y


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    x = np.zeros((total_sample_num, 5))  # 预先分配一个形状为 (total_sample_num, 5) 的数组
    y = np.zeros((total_sample_num, 5))
    # Y = np.zeros((total_sample_num, 5), dtype=np.float32)  # Y需要是float32类型，因为CrossEntropyLoss期望float输入
    for i in range(total_sample_num):
        xp, yp = build_sample()
        x[i] = xp
        y[i] = yp
    return torch.from_numpy(x).float(), torch.from_numpy(y).float()  # 直接从NumPy数组创建张量


def count_class_samples(y):
    class_counts = np.zeros(5)
    for yp in y:
        class_counts[np.argmax(yp)] += 1
    return class_counts.tolist()


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    class_counts = count_class_samples(y)
    print("本次预测集中共有%d个a类样本，%d个b类样本, %d个c类样本, %d个d类样本, %d个e类样本" % (tuple(class_counts)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        y_pred_class = torch.argmax(y_pred, dim=1)  # 获取预测类别
        y_true_class = torch.argmax(y, dim=1)  # 获取真实类别
        for y_p, y_t in zip(y_pred_class, y_true_class):  # 与真实标签进行对比
            if y_p == y_t:
                correct += 1  # 样本类别判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
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
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [ll[0] for ll in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [ll[1] for ll in log], label="loss")  # 画loss曲线
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
        res_class = np.argmax(res)  # 获取预测类别
        res_prob = res[res_class]
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, res_class, res_prob))  # 打印结果


if __name__ == "__main__":
    main()
    # test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.bin", test_vec)
