import torch
import torch.nn as nn
import numpy as np

# import matplotlib.pyplot as plt

"""
基于pytorch实现一个多分类的机器学习任务
输入为一个五维张量  输出为一个只包含四个0 一个1的五维张量 
输入中最大值所在的索引位置对应输出五维张量中1所在的位置，其余索引位置为0
1所在的位置代表对应类别  按ABCDE顺序


不同激活函数 损失函数试试
"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        # 定义两层线性层  为了更好拟合数据
        self.linear1 = nn.Linear(input_size, 5)
        # 最后输出为一维
        self.linear2 = nn.Linear(5, 1)
        # 不同的激活函数试试分类任务
        # 使用sigmoid最后输出一个0,1之间的概率值
        # self.activation = torch.sigmoid
        # 考虑到sigmoid恒定输出在0，1之间无法拟合真实数据 所以采用relu
        self.activation = torch.relu
        # 定义损失函数 由于只输出一个概率值 所以考虑使用均方差函数
        self.loss = nn.functional.mse_loss
        # 为什么换了CrossEntropyLoss 这个损失函数变成0了
        # self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # 经过两层线性层
        x = self.linear1(x)
        x = self.linear2(x)
        # 再经过激活函数 得出预测值
        pre_y = self.activation(x)
        if y is not None:
            return self.loss(x, y)
        else:
            return pre_y


# 训练逻辑   单个测试样本  五维张量分别代表 ABCDE类
# 建立单个训练数据  数据中并暗含分类规律供机器学习  并让机器学习其中规律
def build_sample():
    x = np.random.random(5)
    # print(x)
    a = np.argmax(x)
    # print(a)
    b = np.argmin(x)
    # print(b)

    # 输出值对应分类关系 输出值为最大值所在的位置
    # if a == 0:
    #     return x, 1 # A类
    # if a == 1:
    #     return x, 2 # B类
    # if a == 2:
    #     return x, 3 # C类
    # if a == 3:
    #     return x, 4 # D类
    # if a == 4:
    #     return x, 5 # E类

    # 搞个符合rule的测试数据再试试
    if a == 0:
        return x, 2  # A类
    if a == 1:
        return x, 4  # B类
    if a == 2:
        return x, 6  # C类
    if a == 3:
        return x, 8  # D类
    if a == 4:
        return x, 10  # E类


def build_dataset(total_sample_num):  # total_sample_num 为想要获得的训练样本的数量
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])  # 这里输出是标量所以需要加[]
    return torch.FloatTensor(X), torch.FloatTensor(Y)


def evaluate(model):
    # 将模型设置为评估模式
    model.eval()
    # 每轮100个数据做一次预测
    test_sample_num = 100
    # 创建100个测试数据
    x, y = build_dataset(test_sample_num)
    # print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    # 模型不做梯度变化
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        # print('----------------------')
        # print(y)
        # print(y_pred)
        # print('----------------------')
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            # 如果最大值在第一位        由于经过sigmod函数输出在0，1之间 所以乘以5来映射测试数据  感觉此处逻辑有问题 感觉是反向操作了，应该选用合适模型去拟合真实数据才对
            # if np.argmax(y_t) == 0 and 0 < 5 * y_p[0] <= 1:
            #     correct += 1
            # elif np.argmax(y_t) == 1 and 1 < 5 * y_p[0] <= 2:
            #     correct += 1
            # elif np.argmax(y_t) == 2 and 2 < 5 * y_p[0] <= 3:
            #     correct += 1
            # elif np.argmax(y_t) == 3 and 3 < 5 * y_p[0] <= 4:
            #     correct += 1
            # elif np.argmax(y_t) == 4 and 4 < 5 * y_p[0] <= 5:
            #     correct += 1
            # else:
            #     wrong += 1
            # 不同预测逻辑，判定什么算预测对           这个是否可以随便定义什么算对什么算错
            # if y_t == 2 and 0 < y_p[0] <= 2:
            #     correct += 1
            # elif y_t == 4 and 2 < y_p[0] <= 4:
            #     correct += 1
            # elif y_t == 6 and 4 < y_p[0] <= 6:
            #     correct += 1
            # elif y_t == 8 and 6 < y_p[0] <= 8:
            #     correct += 1
            # elif y_t == 10 and 8 < y_p[0] <= 10:
            #     correct += 1
            # else:
            #     wrong += 1
            # 不同预测逻辑
            if y_t == 2 and 0 < y_p[0] <= 3:
                correct += 1
            elif y_t == 4 and 3 < y_p[0] <= 5:
                correct += 1
            elif y_t == 6 and 5 < y_p[0] <= 7:
                correct += 1
            elif y_t == 8 and 7 < y_p[0] <= 9:
                correct += 1
            elif y_t == 10 and 9 < y_p[0]:
                correct += 1
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
    # learning_rate = 0.1  # 学习率
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
    torch.save(model.state_dict(), "model.bin3")
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
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        # print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果
        pre_y = res[0]
        a = ''
        # if 0 < pre_y <= 2:
        #     a = 'A类'
        # if 2 < pre_y <= 4:
        #     a = 'B类'
        # if 4 < pre_y <= 6:
        #     a = 'C类'
        # if 6 < pre_y <= 8:
        #     a = 'D类'
        # if 8 < pre_y <= 10:
        #     a = 'E类'
        if 0 < pre_y <= 3:
            a = 'A类'
        if 3 < pre_y <= 5:
            a = 'B类'
        if 5 < pre_y <= 7:
            a = 'C类'
        if 7 < pre_y <= 9:
            a = 'D类'
        if 9 < pre_y:
            a = 'E类'

        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, a, res))  # 打印结果


if __name__ == '__main__':
    main()
    # a = np.random.random(1)
    # print(a)
    # b = a[0]
    # print(b)

    # x = np.random.random(5)
    # print(x)

    # a = np.arange(10).reshape(5, 2)
    # a[1, 1] = 9
    # print(a)
    # array([[0, 1],
    #       [2, 9],
    #       [4, 5],
    #       [6, 7],
    #       [8, 9]])
    # b = np.argmax(a)
    # print(b)
    # c = np.argmax(a, axis=0)
    # print(c)
    # array([4, 1], dtype=int64),即a[0,4],a[1,1]
    # np.argmax(a, axis=1)
    # array([1, 1, 1, 1, 1], dtype=int64)

    # test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.bin3", test_vec)

    # something = 'something'
    # options = {'this': 1, 'that': 2, 'there': 3}
    # the_thing = options.get(something, 4)
    # print(the_thing)
