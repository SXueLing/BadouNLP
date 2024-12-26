import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


"""
改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。
"""
class ClassfierModule(nn.Module):
    """docstring for Classfier"""
    def __init__(self, input_size):
        super(ClassfierModule, self).__init__()
        self.liner = nn.Linear(input_size,5)
        self.relu = nn.ReLU()

    def forward(self,x)->torch.Tensor:
        # y = self.relu(self.liner(x))
        y = self.liner(x)
        return y

def buile_sample():
    x = np.random.random(5)
    y = np.argmax(x)  # 返回预测的索引
    return x,y

def bulid_dataset(sample_count:int)->tuple[torch.tensor,torch.tensor]:
    X,Y = [],[]
    for _ in range(sample_count):
        x,y = buile_sample()
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return torch.FloatTensor(X),torch.LongTensor(Y)

def evaluate(model:nn.Module):
    model.eval()
    X_test,Y_test = bulid_dataset(100)
    with torch.no_grad():
        predictions = model(X_test)
        predicted_class = torch.argmax(predictions,dim=1) # 每一行挑选最大
        accuracy = (predicted_class == Y_test).float().mean().item()
    print(f"测试集准确率: {accuracy * 100:.2f}%")
    return accuracy

def train():
    # 超参数设置
    epochs = 100
    batch_size = 40
    learn_rate = 0.001
    train_sample = 5000
    input_size = 5

    # 初始化模型，损失函数，优化器
    model = ClassfierModule(input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learn_rate)

    #记录训练日志
    log= []

    # 加载数据
    X_train,Y_train = bulid_dataset(train_sample)

    # 开始分批多轮训练
    for epoch in range(epochs):
        model.train()
        epoch_loss = []
        for i in range(0,train_sample,batch_size):
            x_batch = X_train[i:i+batch_size]
            y_batch = Y_train[i:i+batch_size]

            # 前向转播记录损失
            predictions = model(x_batch)
            loss = criterion(predictions,y_batch)

            # 反向传播进行参数调整
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        # 记录每个epoch的平均损失与准确率
        avg_loss = np.mean(epoch_loss)
        acc = evaluate(model) # 将每一轮训练的参数效果进行展示
        log.append([acc,avg_loss])

        print(f'第{epoch+1}/{epochs}训练中，对应损失为：{avg_loss:.4f},对应准确率：{acc:.4f}')

    # 保存模型
    torch.save(model.state_dict(), "multi_class_classifier.pth")

    plt.plot(range(len(log)), [entry[0] for entry in log], label="Accuracy")
    plt.plot(range(len(log)), [entry[1] for entry in log], label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

# 定义一个测试函数，评价该模型效果


    pass

if __name__ == '__main__':
    train()

    # 模型输出尺寸查看
    # model = ClassfierModule(5)
    # x = torch.rand([2,5],dtype=torch.float32)
    # with torch.no_grad():
    #     y = model(x)
    # print(x)
    # print(y)
    


