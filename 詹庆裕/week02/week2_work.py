import torch
import torch.nn as nn
import numpy as np

# 1.初始化模型
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel,self).__init__()
        self.linear = nn.Linear(input_size,5)   #线性层，输入一个五维向量，输出一个代表五类的向量
        self.activation = torch.softmax    #激活函数
        self.loss = nn.CrossEntropyLoss()   #损失函数 交叉熵

    def forward(self,x, y=None):  #loss值计算函数，同时也能给出预测值
        x = self.linear(x)  #先经过线性层
        if y is not None:
            return self.loss(x, y)  #返回loss值
        else:
            probability_dist = self.activation(x, dim=1)   #获取分布概率
            y_pred = torch.argmax(probability_dist, dim=1)  #取最大数的索引
            return y_pred  #返回预测值


#生成样本
def example():
    x = np.random.random(5)  #生成一个五维向量
    y = np.argmax(x)
    return  x, y


#生成训练集
def example_list(example_num):
    example_x = []
    example_y = []
    for temp in range(example_num):
        x , y = example()
        example_x.append(x)
        example_y.append(y)
    return torch.FloatTensor(np.array(example_x)), torch.LongTensor(np.array(example_y))


# 阶段性评估
def evaluate(model):
    model.eval()  #将模型调为测试模式
    test_example_num = 100   #测试样本量
    correct , wrong = 0 ,0   #记录预测的正确数与错误数
    test_x, test_y = example_list(test_example_num)
    y_pred = model(test_x)
    for y_p, y_t in zip(y_pred, test_y):
        if  y_p.item() == y_t.item():   #预测值与真实值一样时视为预测正确
            correct += 1
        else:
            wrong += 1
    print("本次评估正确个数为:%d，正确率为:%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 训练过程
def main():
    input_size = 5  # 5维向量
    batch_size = 20  # 单次训练的数据量
    example_num = 5000  # 训练样本的数量
    epoch_size = 20  # 训练的次数
    learning_rate = 0.01  # 学习率
    model = TorchModel(input_size)   #初始化模型
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)  #优化器选择
    X ,Y = example_list(example_num)   #获取训练样本集
    loss_date = []  #用于记录每次评估模型的数据
    #开始训练
    for epoch in range(epoch_size):
        model.train()  # 将模型设置为训练模式
        watch_loss = []  # 用于存储每轮的loss
        for epoch_index in range(example_num//batch_size):
            batch_x = X[(epoch_index * batch_size) : ((epoch_index + 1) * batch_size)]
            batch_y = Y[(epoch_index * batch_size) : ((epoch_index + 1) * batch_size)]
            loss = model(batch_x, batch_y)    #获取loss值
            loss.backward()        # 计算梯度
            optim.step()      # 更新权重
            optim.zero_grad()  # 梯度归0
            watch_loss.append(loss.item())  #记录本批次的loss值
        print("第%d轮的loss值为：%f" % (epoch+1, np.mean(watch_loss)))
        #每轮后要对模型进行评估
        acc = evaluate(model)
        loss_date.append([acc, float(np.mean(watch_loss))])
    torch.save(model.state_dict(), "model.pt")

# 模型应用
def predict(model_path, test_set, input_size):
    model = TorchModel(input_size)   #搭建一个预训练状态模型
    model.load_state_dict(torch.load(model_path))   #将参数加载到模型当中
    print(model.state_dict())   #打印各项参数

    model.eval()  #将模型设置为测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(test_set))   #得到预测的类别
    for tes, res in zip(test_set, result):
        print("输入：%s, 预测类别：%d" % (tes, round(float(res))))

if __name__ == '__main__':
    main()
    test_set = [[1.321432, 3.31243243, 2.432543534, 3.4325435, 2.43243234],  #类别3
                [2.432439, 4.32198439, 3.321243244, 0.4362782, 3.48261482],  #类别1
                [3.643278, 1.54375435, 5.402343243, 4.4328743, 1.32874832],  #类别2
                [3.939054, 2.43948324, 3.854392543, 5.3218543, 1.43728435]]  #类别3
    predict("model.pt", test_set, 5)
