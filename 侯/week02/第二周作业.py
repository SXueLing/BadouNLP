"""
@Project ：cgNLPproject 
@File    ：DAY01_05.py
@Date    ：2024/12/3 15:47 
week2作业，在DAY01_04的模型的基础上调整，
从规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本
改成规律：x是一个5维向量，取向量中最大的数值，其位置是则为此向量的类别
"""
import torch
import torch.nn as nn
import numpy as np

class MyTorchModule(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyTorchModule, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.loss = nn.functional.cross_entropy
        # self.activation = nn.Softmax()

    def forward(self, input_x, output_y=None):
        output_y_pred = self.linear(input_x)
        # target_y = self.activation(output_y_pred)
        if output_y is None:
            return output_y_pred
        else:
            return self.loss(torch.FloatTensor(output_y_pred), output_y)

def build_sample(x_size, y_size):
    x = np.random.random(x_size)
    y = list(x)
    return x, y.index(max(y))


def build_dataset(sample_num, x_size):
    X = []
    Y = []
    for i in range(sample_num):
        x_input, target_y = build_sample(x_size, None)
        X.append(x_input)
        Y.append(target_y)
    # print('X:\n', X)
    # print('X:\n', np.array(X).shape)
    # print('Y:\n', Y)
    # print('Y:\n', np.array(Y).shape)
    return X, Y

def evaluate(myModule, input_size):
    sample_num = 100
    eva_x, eva_y = build_dataset(sample_num, input_size)
    correct = 0
    wrong = 0
    with torch.no_grad():
        eva_y_pred = myModule.forward(torch.FloatTensor(eva_x))
        for i, j in zip(eva_y_pred, eva_y):
            i1 = list(i)
            i2 = i1.index(max(i1))
            if i2 == j:
                correct += 1
            else:
                wrong += 1
            # print(f'y_p：{i}, i2:{i2}, y_t：{j}')

    print(f'正确预测：{correct}个, 错误预测：{wrong}个')


def predict(module_path, x):
    input_size = 5
    output_size = 5
    preModule = MyTorchModule(input_size, output_size)
    preModule.load_state_dict(torch.load(module_path, weights_only=False))
    print(preModule.state_dict())

    preModule.eval()
    with torch.no_grad():
        res = preModule.forward(torch.FloatTensor(x))
    for x1, e in zip(x, res):
        e1 = e.tolist()
        print("输入:{},预测:{}".format(x1, e1.index(e.max())))


def main():
    epoch_num = 50
    train_sample_num = 3000
    batch_size = 20
    input_size = 5
    output_size = 5
    lr = 0.001

    myModule = MyTorchModule(input_size, output_size)
    optim = torch.optim.Adam(myModule.parameters(), lr=lr)
    train_X, train_Y = build_dataset(train_sample_num, input_size)

    for epoch in range(epoch_num):
        print("=========")
        myModule.train()
        epoch_loss = []
        for batch_index in range(train_sample_num // batch_size):
            train_X_batch = train_X[batch_index * batch_size: (batch_index + 1) * batch_size]
            train_Y_batch = train_Y[batch_index * batch_size: (batch_index + 1) * batch_size]

            loss = myModule(torch.FloatTensor(train_X_batch), torch.LongTensor(train_Y_batch))
            loss.backward()
            optim.step()
            optim.zero_grad()
            epoch_loss.append(loss.item())
        evaluate(myModule, input_size)
        print(f"第{epoch}轮平均loss:{np.mean(epoch_loss)}")

    torch.save(myModule.state_dict(), 'week02_homework_DAY01_05.bin')



if __name__ == "__main__":
    # x, y = build_dataset(100, 5)
    # main()
    test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("week02_homework_DAY01_05.bin", test_vec)


