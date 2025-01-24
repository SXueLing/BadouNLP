import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

"""
    改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。
"""
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5) #交叉熵的输入维度和输出维度必须一致
        self.lose = nn.functional.cross_entropy
    def forward(self, x,y=None):
        x = self.linear(x)
        if y is not None:
            return self.lose(x,y)
        else:
            return x

#随机生成一个五纬向量,返回带上最大值的下标
def build_sampe():
    x = np.random.random(5)
    return x,np.argmax(x).item()
def build_data(total_samples_num):
    X = []
    Y = []
    for i in range(total_samples_num):
        x,y = build_sampe()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X),torch.LongTensor(Y)

#测试模型
def evaluate(model):
    model.eval()
    x, y_true = build_data(100)
    correct_count = 0
    with torch.no_grad():
        y_pred = model(x)
        # 将y_pred转换为类别索引
        y_pred_indices = torch.argmax(y_pred, dim=1)
        for pred_index, true_index in zip(y_pred_indices, y_true):
            if pred_index == true_index:
                correct_count += 1
    accuracy = correct_count / len(y_true)
    print(f"准确率: {accuracy}")
    return accuracy





def main():
    total_samples_num = 5000 #数据总数
    input_size = 5 #样本维数
    learning_rate = 0.0001 #学习率
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    model = TorchModel(input_size)
    #选择优化器
    optim = torch.optim.Adam(model.parameters(),lr= learning_rate)
    train_x,train_y = build_data(total_samples_num)
    log = []
    for i in range(epoch_num):
        model.train()
        watch_lose = []
        for batch_index in range(total_samples_num // batch_size):
            x =  train_x[batch_index*batch_size:(batch_index+1)*batch_size]
            y_true =  train_y[batch_index*batch_size:(batch_index+1)*batch_size]
            lose = model.forward(x,y_true)
            lose.backward() #计算梯度
            optim.step() #更新权重
            optim.zero_grad()
            watch_lose.append(lose.item())
        print("=========\n第%d轮平均loss:%f" % (i + 1, np.mean(watch_lose)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_lose))])
        # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
        main()













