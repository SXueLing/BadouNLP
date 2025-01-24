
# coding:utf8
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
class MultiClassficationModel(nn.Module):
    def __init__(self, input_size):
        super(MultiClassficationModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  
        self.loss = nn.functional.cross_entropy  

    def forward(self, x, y=None):
        y_pred = self.linear(x)  
        if y is not None:
            return self.loss(y_pred, y)  
        else:
            return y_pred  

def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)
    return x, max_index

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  
        for y_p, y_t in zip(y_pred, y):  
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5  
    learning_rate = 0.001  
    model = MultiClassficationModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    train_x, train_y = build_dataset(train_sample)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  
            loss.backward()  
            optim.step()  
            optim.zero_grad()  
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  
        log.append([acc, float(np.mean(watch_loss))])
    torch.save(model.state_dict(), "model.pt")
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  
    plt.legend()
    plt.show()
    return

def predict(model_path, input_vec):
    input_size = 5
    model = MultiClassficationModel(input_size)
    model.load_state_dict(torch.load(model_path))  
    print(model.state_dict())

    model.eval()  
    with torch.no_grad():  
        result = model.forward(torch.FloatTensor(input_vec))  
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, torch.argmax(res), res))  

if __name__ == "__main__":
    main()
    test_vec = [[0.12345678, 0.87654321, 0.23456789, 0.98765432, 0.54321098],
                [0.90123456, 0.01234567, 0.45678901, 0.32109890, 0.78901234],
                [0.67890123, 0.54321098, 0.01234567, 0.23456789, 0.10123456],
                [0.23456789, 0.78901234, 0.67890123, 0.10123456, 0.87654321]]
    predict("model.pt", test_vec)
