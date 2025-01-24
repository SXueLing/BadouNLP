import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class TorchModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TorchModel, self).__init__()
        self.linear_one = nn.Linear(input_size, 80)
        self.linear_two = nn.Linear(80, 64)
        self.out = nn.Linear(64, output_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y = None):
        x = self.linear_one(x)
        x = self.linear_two(x)
        x = self.out(x)

        if y is not None:
            return self.loss(x, y)
        else:
            return x


def build_sample():
    x = np.random.rand(5)
    y = 0
    max_value = x[0]

    for index in range(len(x)):
        if max_value < x[index]:
            max_value = x[index]
            y = index

    return x, y


def build_dataset(num_of_samples):
    X = []
    Y = []
    for i in range(num_of_samples):
        x, y = build_sample()
        X.append(x)
        Y.append(y)

    return torch.FloatTensor(X), torch.LongTensor(Y)


def evaluate(model):
    model.eval()
    test_sample_num = 150
    X, Y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model.forward(X)

        for y_p, y_t in zip(y_pred, Y):
            _, p = torch.max(y_p, dim=0)

            if p == y_t:
                correct += 1
            else:
                wrong += 1

        accuracy = correct / test_sample_num
        print('Accuracy: ', accuracy)
    return accuracy


def main():
    epoch_num = 20
    num_of_train_samples = 8000
    input_size, output_size = 5, 5
    batch_size = 40
    learning_rate = 0.001
    model = TorchModel(input_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log = []

    train_X, train_Y = build_dataset(num_of_train_samples)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_num in range(num_of_train_samples // batch_size):
            X = train_X[batch_num * batch_size: (batch_num + 1) * batch_size]
            Y = train_Y[batch_num * batch_size: (batch_num + 1) * batch_size]

            optimizer.zero_grad()
            loss = model(X, Y)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), 'model.bin')
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


def predict(model_path, input_vector):
    input_size = 5
    output_size = 5
    model = TorchModel(input_size, output_size)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vector))
        for vector, res in zip(input_vector, result):
            y_pred = torch.max(res, dim=0)[1]
            print("输入：%s, 预测类别：%d" % (vector, int(y_pred)))


if __name__ == '__main__':
    main()
    test_vec = [[0.97889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
                [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.00797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.09349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]]
    predict("model.bin", test_vec)
