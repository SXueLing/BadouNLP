# 代码 
import numpy as np
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

MODEL_PATH = r".\model_test2.pt"


class TorchModule(nn.Module):
    """
    有关于构建一个深度神经网络类型的例子
    1. 输入层、隐藏层、输出层，每层的神经元个数
    2. 层级之间的计算方式 -->如线性计算
    3. 选择神经元的激活函数 -->增加非线性拟合能力
    4. 选择训练方向损失函数  -->促使模型收敛训练工具
    """

    def __init__(self, input_size, center_size, class_size):
        """
        1. 继承nn模块的框架采用super实现
        2. 设置网络的层级与每层神经元个数   ===>  拿到实例化对象深度学习网络
        3. 设置神经元的激活函数如sigmoid   ===>  拿到计算神经元公式的函数地址
        4. 设置损失函数如均方差mse         ===>  拿到计算均方差公式的函数地址
        :param input_size: 输入层神经元个数 ===> 分类任务中是事物的特征个数
        :param center_size: 隐藏层神经元个数 ===> 根据输入层1~1.5倍调整
        :param class_size: 输出层神经元个数 ===> 是对事物分类的类型个数，想分几类就写几个
        """
        super().__init__()
        self.layer1 = nn.Linear(input_size, center_size)
        self.layer2 = nn.Linear(center_size, class_size)
        self.activate = torch.softmax
        self.activate2 = torch.sigmoid
        self.loss_func = nn.functional.cross_entropy

    def forward(self, x, y=None):
        """
        主要功能：向前计算公式
        1. 输入数据通过线性层公式计算传入下一层  ===>  调用layer对象计算下一层的输入
        2. 拿到当前层级输入计算神经元作为下一层线性计算   ===>  调用activate激活函数计算
        3. 返回输出层的输出值作为模型预测值,或者将损失值当做返回值  ===> 调用loss损失函数计算差异值
        :param x: 输入真实的样本数据
        :param y: 设置一个开关，当为空的时候，不计算损失函数，输出模型预测值，当不为空，计算损失函数并输出损失值
        :return: 模型输出的预测值
        """
        x = self.layer1(x)
        x = self.activate2(x)
        x = self.layer2(x)
        y_predi = self.activate(x, dim=1)
        if y is not None:
            return self.loss_func(y_predi, y)
        else:
            return y_predi


def build_dataset(test=False):
    """
    我们这里直接采用sklean中的花的特征数据
    一共统计了150组数据，每种花有50组数据
    每个花都有四个特征值，其中只有三种花
    我们采用np.random.shuffle(sample)将数据进行打乱
    :return: 输入与输出张量元组
    """
    X = []
    y = []
    X_c, y_c = load_iris(return_X_y=True)
    sample = np.array(list(range(len(X_c))))
    np.random.shuffle(sample)
    for i in sample:
        X.append(X_c.tolist()[i])
        y.append(y_c.tolist()[i])
    split_point = len(X) // 3 * 2
    if not test:
        return torch.FloatTensor(X), torch.LongTensor(y)
    else:
        return torch.FloatTensor(X[split_point:]), torch.LongTensor(y[split_point:])


def evaluate(model):
    """
    模型的测试流程：  ==>  首先要将模型调成评价测试模式
    1. 产生测试数据样本，得到样本的输入与输出值
    2. 调用模型前向计算，得到模型的预测值
    3. 比对模型的预测值与样本输出值，得出总体的重合率并返回
    :param model: 训练好的模型
    :return: 返回模型预测值与测试数据的重合率
    """
    model.eval()
    test_sample_size = 49
    x, y = build_dataset(test=True)
    # print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_size - sum(y)))
    y = y.tolist()
    correct, wrong = 0, 0
    with torch.no_grad():
        y_predi = model(x).numpy().tolist()
        y_predi_max = np.max(model(x).numpy(), axis=1).tolist()
        for i in range(len(y_predi)):
            classes = list.index(y_predi[i], y_predi_max[i])
            print("y预测值：", y_predi[i], "其中最大值: ", y_predi_max[i],
                  "其中求得索引为： ", classes, "正确的所以是：", y[i])
            if classes == y[i]:
                correct += 1
            else:
                wrong += 1
        print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
        return correct / (correct + wrong)


def main_run():
    """
    主要的模型的训练过程：
    1. 训练的必要配置参数设置  ==> 写在前面方面我们去调整训练的细节
        a. 迭代次数
        b. 每次迭代中更新一次梯度的训练的样本数
        c. 训练样本的数据量
        d. 输入的向量数据的维度、以及每层的维度
        e. 训练的学习率
    2. 建立模型  ==> 调用我们定义的网络类实例化一个对象,
        此对象可以被当做函数调用，默认调用对象下的forward方法
    3. 选择优化器  ==> 就是反向传播需要的模型参数的更新公式，如SGD梯度下降
        需要传入模型的初始化参数w，b，和训练学习率
    4. 产生样本点的输入与输出  ==> 主要是tensor张量的形式，一般我们应该自己获取样本数据
    5. 训练迭代     ==> 单次训练迭代的过程，首先将模型调成训练模式
        5.1. 模型前向计算，拿到损失函数的值
        5.2. 模型反向传播计算，计算梯度后更新参数
        5.3. 重置梯度，我们发现都是一大块一大块样本进行训练，都是累计梯度，要归零
        5.4. 记录损失函数值日志
    6. 测试训练好的模型  ==> 获取模型测试的正确率，并记录日志，方便查看模型的测试情况
    """
    # 1. 参数配置
    iter_time = 1000
    batch_size = 5
    train_sample_size = 140
    input_size = 4
    hidden_size = 4
    output_size = 3
    learning_rate = 0.005
    # 2. 建立模型
    model = TorchModule(input_size, hidden_size, output_size)
    # 3. 优化器设置
    optional = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []  # 记录
    # 4. 输入样本点的输入与输出
    train_x, train_y = build_dataset()
    # 5. 训练迭代
    for epoch in range(iter_time):
        model.train()
        watch_loss = []  # 记录每一代的损失函数的值，可以观测模型是否收敛
        for batch_i in range(train_sample_size // batch_size):
            """
            1. 模型前向计算，拿到损失函数的值
            2. 模型反向传播计算，计算梯度后更新参数
            3. 重置梯度，我们发现都是一大块一大块样本进行训练，都是累计梯度，要归零
            4. 记录损失函数值日志
            """
            x = train_x[batch_i * batch_size:(batch_i + 1) * batch_size]
            y = train_y[batch_i * batch_size:(batch_i + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optional.step()
            optional.zero_grad()
            watch_loss.append(loss.item())
        print("\n第{}轮平均loss:{}".format(epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型   ==> 保存的是就是模型的参数矩阵
    torch.save(model.state_dict(), MODEL_PATH)
    # 模型训练可视化 ==> 主要展示训练的收敛情况与测试的准确性
    # print(log)
    plt.rcParams["font.family"] = "FangSong"
    plt.plot(range(len(log)), [l[0] for l in log], color='r', label="accuracy_rate")
    plt.plot(range(len(log)), [l[1] for l in log], color='g', label="loss_value")
    plt.xlabel("iter time")
    plt.legend()
    plt.savefig(r".\train_effect_diagram.jpg", bbox_inches='tight', pad_inches=0)
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def predict(model_path, input_vec):
    """
    1. 首先我们存入的是模型的权重与偏置参数，我们先知道模型的网络结构
    2. 将保存的参数加载到对应网络结构的模型之中，就得到了训练模型
    3. 以模型的评价测试模式就可以通过前向计算得到模型的预测值
    :param model_path: 模型参数的保存路径
    :param input_vec: 模型的样本输入 ==> 必须是张量
    """
    input_size = 4
    hidden_size = 4
    output_size = 3
    model = TorchModule(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    # weights_only=True关键字参数只加载模型的参数，不加载其他模型对象，
    # 这样做是为什么呢？主要因为torch采用的是pickle来保存数据，那么就可以保存python的任意类型，
    # 这样会加载一些我们不需要的内存数据，导致变量可能会被覆写，有数据安全隐患

    with torch.no_grad():
        result = model.forward(input_vec)
        result = result.tolist()
    for i in range(len(input_vec)):
        res = result[i].index(max(result[i]))
        print("输入：%s, 预测类别：%d" % (input_vec[i].numpy().round(2), int(res)))  # 打印结果


if __name__ == '__main__':
    main_run()
    input_diy = build_dataset()[0][20:60]
    # print("\n{}\n测试样本如下：".format("".center(50, '=')), input_diy, sep='\n')
    predict(MODEL_PATH, torch.FloatTensor(input_diy))
    # X = []
    # y = []
    # X_c, y_c = load_iris(return_X_y=True)
    # sample = np.array(list(range(len(X_c))))
    # np.random.shuffle(sample)
    # for i in sample:
    #     X.append(X_c.tolist()[i])
    #     y.append(y_c.tolist()[i])
    # for i, j in zip(X, y):
    #     print(i, j)
    # print(type(X[0]), type(y[0]))
