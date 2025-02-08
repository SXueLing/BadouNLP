import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt



class TestModel(nn.Module):
    def __init__(self, input_size,hidden_size1):
        super(TestModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1) #输入1x5  输出1x5  中间还需要一个5x5
        self.linear2 = nn.Linear(hidden_size1, 5)
        self.loss = nn.CrossEntropyLoss()  # loss采用交叉熵 交叉熵默认做了softmax 不用再激活了？  不确定

    def forward(self, x, y=None):
        x = self.linear1(x)
        y_pred = self.linear2(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


#准备随机训练数据
def build_sample():
    x = np.random.random(5)
    for y,i in enumerate(x): # 找最大值索引
        if i == max(x):
            return x,y

#生成张量数据
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)  #张量数据类型不同

#测试代码
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            a, b = torch.max(y_p,0)  #找预测值的最大值索引
            if b == y_t:
                correct += 1
            else:
                wrong += 1
        acc = correct / test_sample_num
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    hidden_size1 = 5 #中间层维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TestModel(input_size,hidden_size1)
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
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    hidden_size1 =5
    model = TestModel(input_size,hidden_size1)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        y_pre=torch.max(res, dim=0)[1]
        print("输入：%s, 预测类别：%d" % (vec,int(y_pre)))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.97889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
                [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.00797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.09349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]]
    predict("model.bin", test_vec)


