import torch
import torch.nn as nn
import numpy as np

"""
基于pytorch框架编写模型训练
实现一个 寻找 近几年 居住时间 最长的城市
规律：x是一个5维向量，每个值代表 城市居住的天数 ，城市分别为 北京、上海、广州、深圳、杭州
如: [29,13,32,31,45] 代表居住时间最长的城市为 杭州（45天）
"""

# 定义城市居住计算模型
class LiveCityModel(nn.Module):
    # city_num 统计城市的数量 （标签个数）
    def __init__(self, city_num):
        super(LiveCityModel, self).__init__()
        # 使用线性模型
        self.linear = nn.Linear(city_num, 5)
        # 定义 激活函数 Softmax
        self.activation = nn.functional.softmax
        # 定义损失函数 为交叉商函数
        self.loss = nn.functional.cross_entropy

    # 当输入实际值，返回loss值；无实际值，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)
        # softMax 需要设置 维度值 此处为 1
        pred = self.activation(x, dim=1)
        if y is None:
            return pred
        else:
            return self.loss(pred, y)


# 生成测试数据
def generate_data():
    # x = np.random.randint(low=0, high=100, size=5)
    # 样本数据好像也不能过大 不然计算会出错
    x = np.random.random(5)
    # 构造y 为一个 都为 0 的向量
    y = np.zeros_like(x)
    # 将 y 脚标 等于 x中最大值的脚标 的值设置为 1
    y[np.argmax(x)] = 1
    return x, y

# 生成数据集
def generate_dataset(size):
    X,Y = [],[]
    for i in range(size):
       x,y =  generate_data()
       X.append(x)
       Y.append(y)
    return torch.FloatTensor(X),torch.FloatTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = generate_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if np.argmax(y_p.numpy())  == np.argmax(y_t.numpy()):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)



# 训练方法
def train():
    epochs = 100 # 训练的轮次
    batch_size = 20 # 批次大小
    data_size = 5000 # 单次训练的数据量
    learning_rate = 0.001 # 学习率
    model = LiveCityModel(5)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        # 设置模型为训练模式
        model.train()
        # 生成数据集
        train_x,train_y = generate_dataset(data_size)
        watch_loss = []
        for batch_index in range(data_size // batch_size):
            x = train_x[batch_index*batch_size:(batch_index+1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            # 计算loss 值
            loss = model.forward(x,y)
            # 计算梯度
            loss.backward()
            # 重新计算权重
            optim.step()
            # 梯度归零
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        evaluate(model)

if __name__ == '__main__':
    train()
