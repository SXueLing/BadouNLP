# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，预测哪个位置的数值最大

"""

class TorchModel(nn.Module):
    """
    定义神经网络模型
    """
    def __init__(self, input_size, output_size=5):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  # 线性层，输入维度为input_size，输出维度为output_size
        self.activation = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        """
        前向传播
        :param x: 输入数据
        :param y: 真实标签（可选）
        :return: 如果提供了y，则返回损失值；否则返回预测结果
        """
        x = self.linear(x)  # 线性变换
        if y is not None:
            # 计算交叉熵损失，这里不需要再应用softmax，因为cross_entropy函数内部会处理
            return nn.functional.cross_entropy(x, y)
        else:
            # 应用softmax激活函数，以便得到概率分布
            return self.activation(x)

# 生成一个样本
def build_sample():
    """
    随机生成一个5维向量
    :return: 样本向量和对应的标签
    """
    x = torch.randn(5)
    max_index = x.argmax()
    y = torch.zeros_like(x)
    y[max_index] = 1
    return x, y

# 随机生成一批样本
def build_dataset(total_sample_num):
    """
    生成训练数据集
    :param total_sample_num: 总样本数量
    :return: 特征数据和标签数据
    """
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.stack(X), torch.stack(Y)

# 测试代码
def evaluate(model):
    """
    评估模型性能
    :param model: 模型
    :param X: 测试集特征
    :param Y: 测试集标签
    :return: 准确率
    """
    model.eval()  # 将模型设置为评估模式
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    correct = 0  # 初始化正确预测的数量
    total = 0  # 初始化总预测的数量
    with torch.no_grad():  # 禁用梯度计算，评估时不需要计算梯度
        y_pred = model(x)  # 使用模型对测试集X进行预测，得到预测结果Y_pred
        _, predicted = torch.max(y_pred, 1)  # 在预测结果Y_pred中找到概率最大的索引，即模型预测的类别
        total = y.size(0)  # 获取测试集的样本总数
        correct += (predicted == y.argmax(dim=1)).sum().item()  # 比较预测的类别和真实类别，累加正确预测的数量
    print("正确预测个数：",correct,"正确率：",correct/total)
    return correct / total  # 返回准确率，即正确预测的数量除以总预测的数量

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    """
    使用训练好的模型进行预测
    :param model_path: 模型权重文件路径
    :param input_vec: 输入向量
    """
    # 初始化模型
    input_size = 5  # 输入向量维度
    output_size = 5  # 输出向量维度，即分类数
    model = TorchModel(input_size, output_size)

    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))  # 加载训练好的权重
    model.eval()  # 将模型设置为评估模式

    # 对输入向量进行预测
    with torch.no_grad():  # 禁用梯度计算
        result = model(input_vec.unsqueeze(0))  # 模型预测，添加batch维度,将其从[5]变为[1, 5]
        # 由于result是logits，我们需要应用softmax来获取概率分布
        probabilities = torch.nn.functional.softmax(result, dim=1)
        _, predicted = torch.max(probabilities, 1)  # 获取概率最大的索引，即模型预测的类别

    # 打印预测结果
    print(f"输入向量：{input_vec}")
    print(f"预测类别：第{predicted.item()+1}个数字最大")
    print(f"预测概率分布：{probabilities.squeeze(0).tolist()}")  # 将概率分布转换为列表并打印


def main():
    """
    主函数，训练模型并评估性能
    """
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率

    model = TorchModel(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 使用Adam优化器
    log = []

    train_x, train_y = build_dataset(train_sample)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            optimizer.zero_grad()
            loss = model(x, y)  # 计算预测值与真实值经过损失函数计算得到的损失值
            loss.backward()     # 梯度计算，求导
            optimizer.step()    # 更新权重
            optimizer.zero_grad()   # 梯度归零
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss: {np.mean(watch_loss):.4f}")
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), "model.bin")
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # main()
    X = torch.randn(5)
    print(X)
    predict('model.bin', X)
