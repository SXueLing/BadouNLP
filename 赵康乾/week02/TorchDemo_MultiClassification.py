# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset

'''
torch实现一个简单的多分类任务，包括：
1，规定一个多分类任务，构造一系列x,y真值用于训练
2，构造一个简单的神经网络模型
3，训练模型并打印loss和准确度曲线
4，应用模型进行一次分类预测
'''

'''
1, 5分类任务，输入是一个（10，）的张量，两个一组共5组，差最大的序号代表分类：
[1,2, 4,3, 7,9, 8,2, 5,1],差最大的是（8-2），因此分类数是3
'''

def BuildSample():
    x = np.random.random(10)
    diff_x = [x[i] - x[i+1] for i in range(0, 10, 2)]
    y = np.argmax(diff_x)
    return x,y
def BuildDataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x,y = BuildSample()
        X.append(x)
        Y.append(y) # [y]
    return torch.FloatTensor(X), torch.LongTensor(Y)

'''
2,定义一个3层的简单神经网络，前两层是全连接层+ReLU，输出是全连接层+Softmax
'''
class MultiClassification(nn.Module):
    def __init__(self, input_size = 10, hidden_size_1 = 10, hidden_size_2 = 8, \
                 output_size = 5):
        super(MultiClassification,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1,hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2,output_size)
        #self.softmax = nn.Softmax(dim = 1)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        y_pred = self.fc3(x)
        return y_pred

'''
3,定义一个计算预测值准确度的函数，应用于每轮模型训练之后
'''
def accuracy(preds, labels):
    '''
    preds: 张量，x计算出的预测值
    labels: 张量，对应的真值
    return: 准确率
    '''
    score, pred_classes = torch.max(preds, 1) #在第二个维度上返回最大值及其序号（batch_size, num_classes)
    acc_num = (pred_classes == labels).sum().item() #逐项对比，并累加True的数量，即正确个数
    return acc_num / labels.size(0)

'''
模型训练时，需要定义：
1）训练轮数，
2）每轮使用随机的mini_batch进行训练，需要指定batch_size,
3) loss function和Optimizer
4) 学习率
'''

def main():
    total_sample_num = 5000
    epoch_num = 30
    batch_size = 20
    learning_rate = 0.001

    multi_class_model = MultiClassification()
    criterion = nn.CrossEntropyLoss()
    optimizier = torch.optim.Adam(multi_class_model.parameters(), lr = learning_rate)

    X,Y = BuildDataset(total_sample_num)
    train_set = TensorDataset(X, Y)

    log = []
    for epoch in range(epoch_num):
        multi_class_model.train()
        train_loader = DataLoader(train_set, batch_size, shuffle = True) #每轮都随机打乱数据集
        watch_loss = [] #记录当轮每个mini_batch的loss
        running_train_acc = 0 #累加每轮期间的准确度
        for inputs, labels in train_loader:
            optimizier.zero_grad()
            preds = multi_class_model(inputs) #自动调用forward()
            loss = criterion(preds, labels)
            loss.backward()
            optimizier.step()
            watch_loss.append(loss.item())
            running_train_acc += accuracy(preds, labels)

        mean_loss = float(np.mean(watch_loss))
        #每轮训练完计算train的平均准确度，在新建一个测试集测试模型准确度
        mean_train_acc = running_train_acc / len(train_loader)

        X_Test, Y_Test = BuildDataset(100)
        multi_class_model.eval()
        with torch.no_grad(): #测试中不需要反向传播，不需要计算梯度
            Preds_test = multi_class_model(X_Test)
            test_acc = accuracy(Preds_test, Y_Test)

        #记录当轮的loss和acc，打印信息
        log.append([mean_loss, mean_train_acc, test_acc])
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, mean_loss))

    torch.save(multi_class_model.state_dict(), 'multi_class_model.pt')
    plt.plot(range(len(log)), [l[0] for l in log], label="mean_loss")
    plt.plot(range(len(log)), [l[1] for l in log], label="train_acc")
    plt.plot(range(len(log)), [l[2] for l in log], label="test_acc")
    plt.legend()
    plt.show()
    return


'''
加载训练好的模型，并进行预测
'''

if __name__ == "__main__":
    main()
    test_x, test_y = BuildDataset(1)
    test_model = MultiClassification()
    test_model.load_state_dict(torch.load("multi_class_model.pt"))
    test_model.eval()
    with torch.no_grad():
        result = test_model(test_x)
        predicted_class = torch.argmax(result, dim=1).item()
        print("输入：%s, 预测类别：%d, 概率值：%f, 真值：%d" % (test_x.numpy(), predicted_class, result[0][predicted_class], test_y.item()))
