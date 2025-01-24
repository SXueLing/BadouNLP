import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

#五分类任务，假设预测值为[0.1, 0.2, 0.3, 0.2, 0.1] ,由于0.3最大，所以返回下标2。

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size) #线性层
        self.activation = torch.sigmoid
        # self.activation2 = torch.
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x, y = None):
        x = self.linear1(x)  # (batch_size, input_size) -> (batch_size, batch_size)
        y_pred = self.activation(x) # (batch_size, 1) -> (batch_size, batch_size)
        # print('xxxx',x) 2 * 5矩阵
        # print('yyyy', y)
        # print('y_pred', y_pred)
        # pred = torch.FloatTensor([[0.3, 0.1, 0.3],
        #                   [0.9, 0.2, 0.9],
        #                   [0.5, 0.4, 0.2]])
        # target = torch.LongTensor([1,2,0])
        # print('lossss', self.loss(pred, target))
        # print('loss', self.loss(y_pred, y))
        # return
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_sample():
    x = np.random.random(5)
    max_index = 0
    for i in range(1, len(x)):
        if x[i] > x[max_index]:
            max_index = i
    return x, max_index
    

def build_dataset(total_sample_num):
    X= []
    Y= []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # Y存放实际值的下标，必须是整数
    return torch.FloatTensor(X), torch.LongTensor(Y)

# testX, testY = build_dataset(2)
# testModel = TorchModel(5)
# testModel.forward(testX, testY)
# print('xxxxxxxx', testModel.forward(testX, testY))
    

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x,y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个样本" % (test_sample_num))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x) # model.forward(x)
        for y_p, y_t in zip(y_pred, y): #预测标签，真实标签
            max_index = 0
            for i in range(1, len(y_p)):
                if y_p[i] > y_p[max_index]:
                    max_index = i
            #预测的最大值下标和实际值下标是否相等
            if max_index == int(y_t):
                correct+=1
            else:
                wrong+=1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    epoch_num=60 #训练轮数
    batch_size=20 #每次训练样本个数
    train_sample=1000 # 每轮训练总共训练的样本总数
    input_size=5  # 输入向量维度
    learn_rate=0.03 #学习率
    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr = learn_rate) #选择优化器
    log=[]
    train_x, train_y = build_dataset(train_sample)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x,y) # 计算loss  model.forward(x,y)
            loss.backward() # 计算梯度
            optim.step() # 更新权重
            optim.zero_grad() # 梯度归零
            watch_loss.append(loss.item())
        print("======第%d轮平均loss:%f"%(epoch+1, np.mean(watch_loss)) )
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model2.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

main()











