import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本

"""

class TorchModel(nn.Module):
    def __init__(self,input_size):
        super(TorchModel,self).__init__()
        self.linear = nn.Linear(input_size,input_size)
        self.activation = torch.sigmoid
        self.loss = nn.CrossEntropyLoss # 函数使用交叉熵函数
    def forward(self,x,y=None):
        x = self.linear(x)
        # print("forward --- 1")
        y_pred = self.activation(x)
        # print("forward --- 2")
        # # y_pred = self.linear(x)
        # # y_pred = x
        # print("y_pred:",y_pred,"\ny:",y)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred

# 生成样本
def build_sample():
    x = np.random.random(5)
    y = np.zeros(5)
    y[np.argmax(x)] = 1
    # return x,y
    return x,np.argmax(x)

# 获取批次样本
def build_dataset(train_sample):
    X = []
    Y = []
    for i in range(train_sample):
        x,y = build_sample()
        # print(":::::::::::::",i,x,np.argmax(x))
        X.append(x)
        Y.append(y)  #这里跟平方差损失函数有所区别，不能加[]
    return torch.FloatTensor(X),torch.LongTensor(Y)


def evaluate(model):
    model.eval()
    test_sample_num = 100
    x,y = build_dataset(test_sample_num)
    # print("本次预测集共有%d个正样本，%d个负样本" % (sum(y),test_sample_num-sum(y)))
    correct,wrong = 0,0
    with torch.no_grad():
        y_pred = model(x)
        for y_p,y_t in zip(y_pred,y):
            if np.argmax(y_p) == y_t:
                correct +=1
            else:
                wrong +=1
    print("正确预测个数：%d，正确率：%f" % (correct,correct/(correct+wrong)))
    return correct / (correct + wrong)


def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 0.01

    model = TorchModel(input_size)

    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    log = []

    train_x,train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index+1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index+1) * batch_size]
            # print(",,,,,,,,",x,y)
            loss = model.forward(x,y)
            # print("loss:",loss)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=======\n第%d沦平均loss：%f" % (epoch + 1 , np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc,float(np.mean(watch_loss))])
    torch.save(model.state_dict(),"model1.in")
    print(log)
    plt.plot(range(len(log)),[l[0] for l in log],label='acc')
    plt.plot(range(len(log)),[l[1] for l in log],label='loss')
    plt.legend()
    plt.show()
    return

if __name__ == "__main__":
    main()
