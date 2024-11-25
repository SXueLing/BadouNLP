import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，求最大值
"""

class torchmodel(nn.Module):
    def __init__(self, input_size):
        super(torchmodel, self).__init__()
        self.linear = nn.Linear(input_size, 100)
        self.linear2 = nn.Linear(100, 5)
        self.activation = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.linear(x)
        x = self.activation(x)
        x = self.linear2(x)
        if y is not None:
            return self.loss(x,y)
        else:
            return x

def build_sample():
    x = np.random.random(5)
    y = np.zeros_like(x)
    y[x.argmax()] = 1
    return x, y

def build_dataest(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x,y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X),torch.FloatTensor(Y)

def test(modle):
    with torch.no_grad():
        test_sample_num = 200
        x,y = build_dataest(test_sample_num)
        correct, wrong = 0, 0
        y_pred = modle.forward(x)
        for y_p, y_t in zip(y_pred, y):
            if y_p.argmax() == y_t.argmax():
                correct += 1  # 负样本判断正确
            else:
                wrong += 1
        print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
        return correct / (correct + wrong)

def main():
    epoch_num = 200  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 训练样本总个数
    input_size = 5  # 输入维度
    learning_rate = 0.01  # 学习率
    model = torchmodel(input_size)  # 建立模型
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 设置优化器
    log = []
    train_x, train_y = build_dataest(train_sample)  # 创建训练集，正常任务是读取训练集
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重
            optimizer.zero_grad()  # 权重归零
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = test(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = torchmodel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        # 打印结果
        print(vec)
        print(res)
        print(f"预测最大项为{res.argmax()+1}")


if __name__ == "__main__":
#    main()
    test_sample_num = 200
    test_vec, y = build_dataest(test_sample_num)
    predict("model.bin", test_vec)
