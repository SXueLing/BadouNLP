import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt


#定义神经网络模型
class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        #super：调用父类方法
        super(TorchModel, self).__init__()
        #input_size定义接收几维向量，num_classes定义输出几维向量，也就是几类
        self.linear = nn.Linear(input_size, num_classes)  # 线性层输出5个类别的分数
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失
    #forward方法定义了向前传播过程，也就是如何根据已有的x来得到y，如果是在预测或评估，那么就不需要人为给y：
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(x, y)  # 计算损失
        else:
            return x  # 输出原始分数（不应用softmax这种激活函数）

#定义一个build_sample函数来构造单个样本，x取随机的五维向量，label（即y）取x最大值的对应的索引值
def build_sample():
    x = np.random.random(5)
    # 找到最大值的索引作为类别标签
    label = np.argmax(x)
    return x, label

#定义一个build_dataset函数来构造数据集，它基于build_sample函数来构造数据集，total_sample_num定义了数据集内包含多少个样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        #将build_sample返回结果赋值给x，y（即x赋值给x，label赋值给y）
        x, y = build_sample()
        #将x和y结果累加到X和Y的列表里，每个x是个五维向量，每个y是个整数
        X.append(x)
        Y.append(y)

    # 将列表X和Y转换为PyTorch张量，并返回。其实就是原本X是个列表，现在数值几乎不变，然后要转换为系统识别的一种数学形式，将它称为张量而已，可以认为不涉及什么其他计算，可能涉及的就是精度转换
    # X张量是一个二维数组，其中每一行是一个五维的样本特征向量。（二维数组，代表由行、列构成；每一行是五维向量，举例：[0.1, 0.4, 0.35, 0.8, 0.05]）
    '''
    X的形式可能为：（此时包含4个特征向量，就代表total_sample_num给的是4）
    [0.1, 0.4, 0.35, 0.8, 0.05],  # 第一个样本的特征向量
    [0.2, 0.1, 0.9, 0.25, 0.5],   # 第二个样本的特征向量
    [0.7, 0.6, 0.1, 0.0, 0.6],    # 第三个样本的特征向量
    [0.3, 0.9, 0.2, 0.1, 0.5]     # 第四个样本的特征向量
    
    此时Y就为：[3, 2, 0, 1]
    '''
    # Y张量是一个一维数组，其中每个元素是一个整数标签（一维数组，代表只有行构成；每一行是个整数。其实也就是只有一个整数）
    #FloatTensor用于存储浮点型张量，LongTensor用于存储64位长整型张量
    return torch.FloatTensor(X), torch.LongTensor(Y)

#评估模型的函数
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct = 0
    #下面这个上下文管理器用于禁用梯度计算，这对于推理或评估模型是必要的，因为它可以减少内存消耗并加速计算
    with torch.no_grad():
        y_pred = model(x)  # 通过模型预测，输入x，来输出y，就得到了y的预测值
        #对于y_pred张量的每一行（即每个样本的预测分数），找到该行中的最大值及其索引，由于只要索引，因此最大值用开头的_接收
        _, predicted = torch.max(y_pred, 1)  # 获取预测类别的索引
        #比较预测类别predicted和真实标签y，生成一个布尔型张量，其中True表示预测正确，False表示预测错误。然后，使用.sum()函数计算True的总数，即正确预测的样本数。最后，.item()将结果转换为Python标量
        correct = (predicted == y).sum().item()
    accuracy = correct / test_sample_num
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy

#模型训练的主函数
def main():
    # 设置训练参数
    epoch_num = 20  # 训练的总轮数，即数据集将被完整遍历的次数
    batch_size = 20  # 每一批处理的样本数量
    train_sample = 5000  # 训练集中的总样本数量
    input_size = 5  # 每个样本的输入特征数量，即定义接收几维向量，比如(1,2,3,4,5)就是一个五维向量
    num_classes = 5  # 分类任务中的类别数量
    learning_rate = 0.001  # 学习率，用于控制权重更新的步长

    #定义model模型对象，类为一开始的TorchModel，
    model = TorchModel(input_size, num_classes)

    '''
    下方optim是在定义优化器
    torch.optim.Adam 是 PyTorch 中的一个优化器类，它实现了 Adam 算法
    model.parameters() 是一个nn.Module中的方法，作用是将模型参数传递给优化器Adam
    lr是学习率参数
    '''
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    #这里是要训练5000个样本到数据集中，并且返回最终转换成的X张量和对应的Y标签信息
    #转换后的 train_x 是一个形状为 (5000, 5) （即5000行，5列，这个形式代表二维）的浮点类型张量，它包含了5000个样本，每个样本都是一个五维的向量。而 train_y 是一个形状为 (5000,) （表示一个包含5000个整数的数组，这个形式代表一维）的长整型张量，它包含了与 train_x 中每个样本相对应的标签。
    train_x, train_y = build_dataset(train_sample)

    #epoch即训练，epoch_num在上方定义为20，即训练20次
    for epoch in range(epoch_num):
        #model.train()代表将model模型设置为训练模式
        model.train()
        # 初始化一个列表来存储每个batch（即每个训练批次）的损失值
        watch_loss = []
        #train_sample // batch_size会得到要训练多少批的数据，//是代表取整除结果，即丢弃余数，只取整除的结果，比如1024//20等于51
        for batch_index in range(train_sample // batch_size):
            # 从train_x和train_y中取出当前批次的样本和标签，这里其实是在对train_x和train_y做切片，然后将部分值传给x和y
            #简单来说，假如有一个列表 aa = [5, 4, 3, 2, 1]，并且执行 x = aa[1:3]，那么 x 将会是一个包含从索引 1 开始到索引 3 之前（即不包含索引 3）的所有元素的新列表。因此，x 的值将会是 [4, 3]。
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            # 将当前批次的样本和标签送入模型计算损失
            loss = model(x, y)
            # 清空之前的梯度
            optim.zero_grad()
            # 反向传播计算梯度
            loss.backward()
            # 使用优化器更新模型的权重
            optim.step()
            # 将当前批次的损失值添加到列表中
            watch_loss.append(loss.item())
        # 打印当前epoch的平均损失值
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
def predict(model_path, input_vecs):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # 加载训练好的权重，确保在CPU上加载
    print(model.state_dict())  # 可选：打印模型参数

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        # 将输入向量转换为PyTorch张量，并添加一个新的维度以匹配模型的输入要求（batch_size, input_size）
        input_tensor = torch.FloatTensor(input_vecs)
        result = model(input_tensor)  # 模型预测

    # 对于每个输入向量和对应的预测结果
    for vec, res in zip(input_vecs, result):
        # 找到具有最高分数的类别的索引
        predicted_class = torch.argmax(res, dim=0).item()
        # 如果需要，可以计算softmax来获得概率分布（但这里只打印预测的类别和索引为predicted_class的分数）
        # probabilities = torch.softmax(res, dim=0)
        # print("Probabilities:", probabilities.numpy())

        # 打印结果，将vec转换为列表以便打印
        print("输入：%s, 预测类别：%d, 分数：%f" % (vec, predicted_class, res[predicted_class].item()))


if __name__ == "__main__":
    main()  # 运行主函数进行训练
    # 注意：这里应该使用与保存时相同的文件扩展名，例如 "model.bin" 或 "model.pt"
    test_vec = [
        [0.97889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
        [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
        [0.00797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
        [0.09349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]
    ]
    # 确保文件路径与保存时一致
    predict("model.bin", test_vec)  # 使用训练好的模型进行预测
