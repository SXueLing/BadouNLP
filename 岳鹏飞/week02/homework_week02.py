import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TorchNN(nn.Module):
    """建立一个一层线性曾和一个损失函数为交叉熵（包含softmax激活函数）的神经网络"""
    def __init__(self, input_size, output_size):
        super(TorchNN, self).__init__()
        self.linear = nn.Linear(input_size, output_size) #输入 5 维 ，输出5维
        self.loss = nn.functional.cross_entropy #设置损失函数为交叉熵

    def forward(self, x, y=None):
        y_pred = self.linear(x) # zhi
        if y is not None:
            return self.loss(y_pred, y) #计算预测值和真实值的误差
        else:
            return y_pred

def generate_random_arr(size = 5):
    """生成5个 0 ～ 1 之间的随机数
    in_args
        size 生成样本的维度值，默认为5维向量
    out_args
        numpy() 5个 0 ～ 1之间的随机数的数组"""
    return np.random.rand(size)

def generate_maxnum_mask(arr):
    """生成数组中最大值的位置掩码数组和最大值的数组下标
    如 X = 【1，2，3，4，5】
    则 T = 【0，0，0，0，1】
    in_args
        arr 输入的numpy数组
    out_args
        mask arr中最大值的位置掩码数组
        max_num_index arr数组中最大值的数组下标"""
    #在数组中找到最大的数
    max_num = 0
    for num in arr:
        if num > max_num:
            max_num = num

    # 生成位置掩码
    mask = np.zeros(arr.shape)
    max_num_index = np.where(arr == max_num)[0]
    mask[max_num_index] = 1
    return mask, max_num_index

def generate_train_data(howmany):
    """生成howmany个样本数据
    in_args
        howmany 生成的数据量
    out_args
        tensor(x) 样本点的tensor向量
        tensor(t) 样本点中最大值的位置掩码的tensor向量"""
    X = []
    T = []
    for i in range(howmany):
        x = generate_random_arr()
        X.append(x)
        T.append(generate_maxnum_mask(x)[0])

    return torch.FloatTensor(X), torch.FloatTensor(T)

def evaluate(model):
    """评估每一次训练的准确率
    in_args
        model 模型实例
    out_arg
        ratio 预测的正确率"""
    model.eval()
    test_sample_nums = 100
    x, t = generate_train_data(test_sample_nums)
    correct, incorrect = 0, 0
    delta = 1e-7  # 设置一个微小值，避免除零错误
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, t): #将预测值的位置下标和正确的位置下标进行对比来统计准确率
            if generate_maxnum_mask(y_p)[1] == generate_maxnum_mask(y_t)[1]:
                correct += 1
                # print("evaluate y_p : ", y_p)
            else:
                incorrect += 1
    ratio = correct / (correct + incorrect + delta)
    print("评估数据的数量为:%d 正确的预测数量为:%d 正确率:%f"% (test_sample_nums, correct, ratio))
    return ratio


def main():
    """处理主要的训练逻辑"""
    epoch_times = 200 #训练的总次数
    batch_size = 20 #每次模型处理的样本数量
    samples_per_epoch = 5000 #每次训练的样本总数
    input_dim = 5 #输入模型的样本格式： 5维向量
    output_dim = 5 #预测结果的样本格式： 5维向量
    learning_rate = 0.001 #学习率

    #建立模型
    model = TorchNN(input_dim, output_dim)

    #设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = [] #记录日志

    #准备训练数据
    train_x, train_y = generate_train_data(samples_per_epoch)
    print("train_x : ", train_x)
    print("train_y : ", train_y)

    #开始训练
    for epoch in range(epoch_times):
        model.train()
        watch_loss = []
        for batch_idx in range(samples_per_epoch // batch_size):
            x = train_x[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            t = train_y[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            loss = model.forward(x, t) # 计算交叉熵
            loss.backward()  #计算梯度
            optimizer.step()  #更新权重
            optimizer.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("---------\n第%d轮的平均loss ： %f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  #测试本轮的训练结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "torchNN.pt")
    # 汇总训练信息
    print(log)
    plt.plot(range(len(log)), [x[0] for x in log], label="acc")
    plt.plot(range(len(log)), [x[1] for x in log], label="loss")
    plt.legend()
    plt.show()
    return

def predict(NN, test_data):
    """使用训练好的模型进行预测
    in_arg
        model  训练好的模型
        test_data 准备好的训练数据
    out_arg
    """
    input_size = 5
    output_size = 5
    model = TorchNN(input_size, output_size) # 模型实例化
    model.load_state_dict(torch.load(NN)) # 加载模型权重
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(test_data)) #预测数据

    for vec, res in zip(test_data, result):
        # print("测试数据:" + str(vec) + "正确结果:" + "预测结果:" + str(res))
        index_t = generate_maxnum_mask(np.array(vec))[1] # 注意！vec是以数列的形式传入的，而generate_maxnum_mask处理的是np数组，这里做array的转换，否则会报错
        index_p = generate_maxnum_mask(res.numpy())[1] # ???不知道为什么这个下标有时候打印为空
        print("测试数据:" + str(vec) + "正确结果:" + str(index_t) + "预测结果:" + str(res) + "预测最大值位置:" + str(index_p))

if __name__ == '__main__':
    # main()
    eva_data = [[0.7401, 0.9956, 0.5488, 0.4796, 0.8956],
            [0.4976, 0.7535, 0.9565, 0.1158, 0.4502],
            [0.8447, 0.4343, 0.4277, 0.4793, 0.7973],
            [0.9301, 0.5578, 0.5536, 0.2776, 0.6277],
            [0.5255, 0.1379, 0.0999, 0.3824, 0.7900]]
    predict("torchNN.pt", eva_data)


##### debug code #####
# X = generate_random_arr()
# print("随机的数组为： ", X)
# # T, index = generate_maxnum_mask(X)
# T = generate_maxnum_mask(X)[0]
# index = generate_maxnum_mask(X)[1]
# print(f"最大值的位置掩码为：{T}, 最大值的位置为：{index}")
# a, b = generate_train_data(5)
# print(f"数据：{a}\n掩码：{b} \n")
