import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，5个数字谁最大，输出5个分类，输出向量中最大数字对应的分类为1，其余为0

"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, input_size) # 输入维度为input_size，输出因为是分类所以跟输出一样
        self.loss = nn.CrossEntropyLoss() # 损失函数 使用交叉熵
        # self.loss = nn.functional.cross_entropy

    def forward(self, x, y = None):
        pred = self.linear(x)
        # pred = self.sigmoid(pred)
        if y is not None: 
            loss = self.loss(pred, y)
            return loss
        else:
            return pred
        
def build_simple():
    x = np.random.random(5)
    y = np.zeros(5)
    y[np.argmax(x)] = 1
    return x, y
    
def build_dataset(totol):
    X = []
    Y = []
    for i in range(totol):
        x, y = build_simple()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)

# 评估模型
def evaluate(model):
    test_sample = 100
    test_x, test_y = build_dataset(test_sample)
    model.eval()
    right, wrong = 0, 0
    
    with torch.no_grad(): # 评估不需要计算梯度
        pred = model(test_x)
        for y_p, y_t in zip(pred, test_y):
            if y_p.argmax() == y_t.argmax():
                right += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (right, right / (right + wrong)))
    return right / (right + wrong)

# 预测
def predict(model_path, input):
    model = TorchModel(5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        result = model(torch.FloatTensor(input))
    for vec, res in zip(input, result):
        print("输入向量：%s, 预测类别：%s,  预测结果：%s" % (vec, torch.argmax(res), res))


def main():
    # 训练参数
    epochs = 20 # 迭代次数
    batch_size = 20 # 每次迭代训练的样本数
    input_size = 5 # 输入维度
    train_sample = 5000 # 训练样本数
    learn_rate = 0.001 # 学习率
    # 初始化模型
    model = TorchModel(input_size)
    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=learn_rate)
    # 训练数据
    train_x, train_y = build_dataset(train_sample)
    # 保存训练日志，绘制曲线
    log = []
    for epoch in range(epochs):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward() # 计算梯度
            optim.step() # 更新参数
            optim.zero_grad() # 清空梯度
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, np.mean(watch_loss)])
    torch.save(model.state_dict(), "model.pth")
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend() # 显示图例
    plt.show()



if __name__ == '__main__':
    main()
    test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
            [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
            [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
            [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.pth", test_vec)
