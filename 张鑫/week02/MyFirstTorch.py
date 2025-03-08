"""
基于pytorch框架编写模型训练
实现一个自行构造的分类任务
规律：x是一个5维向量，哪个数大，就属于哪一类
"""
import torch as torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt


# 模型训练
class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        # input_size输入样本大小, num_classes表示输出的类别数量
        self.linear = nn.Linear(input_size, num_classes)
        # 这里不用使用softmax，cross_entropy自带
        # self.activation = torch.softmax
        # self.activation = nn.functional.softmax
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        # x = self.linear(x)
        # y_pred = self.activation(x, dim=1)
        y_pred = self.linear(x)
        if y is None:
            return nn.functional.softmax(y_pred, dim=1)
        else:
            return self.loss(y_pred, y)


# 使用np现成的api
def build_simple_by_numpy():
    x = np.random.random(5)
    x_max_idx = np.argmax(x)
    y = np.zeros_like(x)
    y[x_max_idx] = 1
    return x, y


def build_simple_by_native():
    x = [random.random() for _ in range(5)]
    x_max_index = x.index(max(x))
    y = [1 if i == x_max_index else 0 for i in range(len(x))]
    return x, y


# 随机生成一组样本
def build_data_sets(simple_num):
    arr_x = []
    arr_y = []
    for i in range(simple_num):
        x, y = build_simple_by_numpy()
        # x, y = build_simple_by_native()
        arr_x.append(x)
        arr_y.append(y)
    arr_x = np.array(arr_x)
    arr_y = np.array(arr_y)
    return torch.FloatTensor(arr_x), torch.FloatTensor(arr_y)


# 使用测试集测试模型
# 用来测试每轮模型的准确率
def evaluate(model):
    total = 100
    x, y = build_data_sets(total)
    correct = 0
    with torch.no_grad():
        y_pred = model.forward(x)
        correct += (torch.argmax(y_pred, dim=1) == torch.argmax(y, dim=1)).sum().item()
    print("正确预测个数：%d, 正确率：%.2f" % (correct, correct / total))
    return correct / total


def main():
    # 训练20轮
    epoch_num = 20
    # 学习率
    lr = 0.001
    # 样本总数量
    simple_size = 50000
    # 样本选择批次大小
    batch_size = 50
    # 测试数据集
    train_x, train_y = build_data_sets(simple_size)
    # 建立模型
    model = TorchModel(5, 5)
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 记录
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(simple_size // batch_size):
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            loss = model.forward(x, y)
            # 反向传播, 注意这里的loss由x → self.linear → loss得来，其中的调用链使得loss对象可以反向找到self.linear计算其梯度值，即导数值
            loss.backward()
            # 使用反向传播更新的梯度值，更新权重weight和bias
            optimizer.step()
            # 梯度归零，避免累积，一批一个梯度计算值
            optimizer.zero_grad()
            # loss记录
            watch_loss.append(loss.item())
        print("=======第%d轮平均loss为%.2f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()


# 使用模型预测
def predict(model_path, input_vec):
    model = TorchModel(5, 5)
    model.load_state_dict(torch.load(model_path, weights_only=False))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        print("输入：%s, 预测类别：%s" % (input_vec.tolist(), torch.argmax(result, dim=1).tolist()))
    return


if __name__ == '__main__':
    main()

    # 使用
    x, y = build_data_sets(10)
    predict("model.bin", x)
