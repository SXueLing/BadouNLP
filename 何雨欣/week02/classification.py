import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class classifyModel(nn.Module,):
    def __init__(self, input_size):
        super(classifyModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


def get_sample():
    x = np.random.random(5)
    # 获取最大值的索引
    max_index = np.argmax(x)
    return x, max_index


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = get_sample()
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)


# 测试代码, 用来测试每轮模型的准确率
def evaluate(model,test_sample_num,device):
    model.eval()
    x, y = build_dataset(test_sample_num)
    x_test = torch.tensor(x, dtype=torch.float32).to(device)
    y_test = torch.tensor(y, dtype=torch.long).to(device)

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x_test)  # 模型预测
        for y_p, y_t in zip(y_pred, y_test):  # 与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    from torch.utils import data
    # 配置参数
    epochs = 100  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    sample_num = 1000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device = torch.device('cpu')#('cuda:0')
    # 建立模型
    model = classifyModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    x_train, y_train = build_dataset(sample_num)
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)

    # 打包tensor
    train_dataset = data.TensorDataset(x_train, y_train)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # shuffle是否对数据进行打乱
    train = list(enumerate(train_dataloader))  # 转换成列表
    # 训练过程
    for epoch in tqdm(range(epochs)):
        model.train()
        watch_loss = []
        for batch_idx in range(len(train)):
            _, (data, target) = train[batch_idx]  # 从当前数据中区分出训练输入数据和标签
            loss = model(data, target).requires_grad_(True).to(device)  # 计算训练损失
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model,100,device)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pt")
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
    model = classifyModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, torch.argmax(res), res))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.4963533, 0.5524256, 0.95758807, 0.65520434, 0.84890681],
                [0.48797868, 0.67482528, 0.13625847, 0.34675372, 0.09871392],
                [0.49349776, 0.59416669, 0.92579291, 0.41567412, 0.7358894]]
    predict("model.pt", test_vec)
#















