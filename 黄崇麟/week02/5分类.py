# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.loss = nn.CrossEntropyLoss()  # 损失函数：交叉熵损失

    def forward(self, x, y=None):
        yp = self.linear(x)  # 线性变换
        if y is not None:
            return self.loss(yp, y)  # 计算损失
        else:
            return yp  # 返回预测结果


def build_sample():
    x = np.random.random(5)  # 随机生成5维向量
    max_index = np.argmax(x)  # 最大值的索引即为标签
    return x, max_index


def build_dataset(total_sample_num):
    X, Y = [], []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.LongTensor(Y).squeeze(1)  # 标签维度转为1
# x,y = build_dataset(1)
# print(f'x:{x},\ny:{y}')

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("测试集样本分布：正样本数=%d，负样本数=%d" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x).argmax(dim=1)  # 预测类别
        for y_p, y_t in zip(y_pred, y):
            if int(y_p) == int(y_t):  # 比较预测和真实标签
                correct += 1
            else:
                wrong += 1
    print("正确预测数：%d, 正确率：%.2f%%" % (correct, 100 * correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 参数配置
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每批样本数量
    train_sample = 5000  # 每轮训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率

    # 创建模型
    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 优化器
    log = []  # 记录训练过程

    # 数据集
    train_x, train_y = build_dataset(train_sample)

    # 训练循环
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 前向传播 + 计算损失
            loss.backward()  # 反向传播
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())

        # 打印和记录
        print("第 %d 轮平均loss: %.6f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model.bin")



    # 绘制训练过程
    plt.plot(range(len(log)), [l[0] for l in log], label="Accuracy")
    plt.plot(range(len(log)), [l[1] for l in log], label="Loss")
    plt.legend()
    plt.show()


def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载权重
    model.eval()

    with torch.no_grad():
        result = model(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        predicted_class = res.argmax().item()  # 获取预测类别
        print("输入向量：%s, 预测类别：%d, 概率分布：%s" % (vec, predicted_class, res.tolist()))


if __name__ == "__main__":
    main()
    test_vec = [[0.07889086,0.88229675,0.31082123,0.00504317,0.98920843],
                [0.74963533,0.88242561,0.95758807,0.05520434,0.94890681],
                [0.00797868,0.688482528,0.13625847,0.34675372,0.19871392],
                [0.09349776,0.58416669,0.92579291,0.41567412,0.13588941]]
    predict("model.bin", test_vec)
