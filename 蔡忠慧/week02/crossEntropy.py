import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

"""
本周作业:改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。


基于pytorch框架编写模型训练
用交叉熵实现一个多分类任务
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果最大的数字在哪维，则这个向量就属于哪一类，并标记为1 其余为0
例如：[1,3,2,4,5]  则为[0,0,0,0,1]

"""

class TorchMoudle(nn.Module):
    #定义构造函数
    def __init__(self,input_size):
        super().__init__()
        self.Linear = nn.Linear(input_size,5) #线性层
        self.activation = nn.Softmax(dim=1)  # 激活函数使用softmax 因为分类任务概率之和为1
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵

    def forward(self,x,y=None):
        x = self.Linear(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred,y)  # 预测值和真实值计算损失
        else:
            return y_pred # 输出预测结果


# 生成一个样本，随机生成一个5纬向量，如果最大的数字在哪维，则这个向量就属于哪一类
def build_sample():
    x = np.random.random(5)   # 随机生成一个五维向量
    # 找出第一个最大值的下标
    max_index = np.argmax(x)
    r_arr = [0,0,0,0,0]
    r_arr[max_index] = 1  # 替换下标值为1
    return x, r_arr

# 随机生成一批样本
def build_dataset( total_sample_num ):
    X = []
    Y = []
    for i in range(total_sample_num):
        x,y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X),torch.FloatTensor(Y)

# 测试模型准确率
def evaluate(model):
    model.eval()  # 状态改为测试
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # print("本次预测集中共有%d个样本" % (sum(y), test_sample_num - sum(y)))
    correct = 0
    wrong = 0
    with torch.no_grad():
        y_pred = model(x)   # 模型预测数据
        for y_p, y_r in zip(y_pred, y):
            # 对比真实和预测结果中概率最大的类别下标
            if  torch.argmax(y_p) == torch.argmax(y_r) :
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 参数配置
    epoch_num = 20    # 训练轮数
    batch_size = 20   # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入输出向量纬度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchMoudle(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集 实际项目中是读取训练集
    train_x , train_y = build_dataset(train_sample)
    # 数据训练
    for epoch in range(epoch_num):
        model.train()  # 模型进入训练状态
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index*batch_size:(batch_index+1)*batch_size]
            y = train_y[batch_index*batch_size:(batch_index+1)*batch_size]
            loss = model(x,y) # 计算loss
            loss.backward()  # 计算梯度
            optim.step()   # 更新权重
            optim.zero_grad()  # 每批次梯度归零
            watch_loss.append(loss.item())
        print("+++++++++++++++++\n第%d轮平均loss：%f" % (epoch + 1,np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(),'model.pt')

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchMoudle(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测

    for vec, res in zip(input_vec, result.tolist()):
        print("输入：%s, 预测下标：%d, 概率值：%s" % (vec, res.index(max(res)), res))  # 打印结果
        # print(vec, res, res)  # 打印结果


if __name__ == "__main__":
    main()
    # test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.pt", test_vec)