import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

# GPU训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 配置参数
epoch_num = 50
batch_size = 50
total_num = 10000
train_num = 8000
valid_num = 1000
input_size = 5
layer1_size = 10
layer2_size = 8
output_size = 5


# 样本生成
def sample_generator():
    x = np.random.random(5)
    y = np.zeros(5)
    y[np.argmax(x)] = 1
    return x, y


# 数据集生成
def dataset_generator(sample_num):
    X = []
    Y = []
    for i in range(sample_num):
        x, y = sample_generator()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X).to(device), torch.FloatTensor(Y).to(device)


# 划分训练集验证集测试集
X_total, Y_total = dataset_generator(total_num)
X_train = X_total[:train_num]
Y_train = Y_total[:train_num]
X_valid = X_total[train_num:train_num + valid_num]
Y_valid = Y_total[train_num:train_num + valid_num]
X_test = X_total[train_num + valid_num:]
Y_test = Y_total[train_num + valid_num:]


# 模型结构
class fully_connected(nn.Module):
    def __init__(self, input_size, layer1_size, layer2_size, output_size):
        super(fully_connected, self).__init__()
        self.layer1 = nn.Linear(input_size, layer1_size)
        self.layer2 = nn.Linear(layer1_size, layer2_size)
        self.layer3 = nn.Linear(layer2_size, output_size)
        self.softmax = nn.functional.softmax
        self.loss = nn.functional.cross_entropy
        self.sigmoid = nn.functional.sigmoid

    def forward(self, x, y=None):
        x1 = self.sigmoid(self.layer1(x))
        x2 = self.sigmoid(self.layer2(x1))
        y_pred = self.softmax(self.layer3(x2), dim=1)
        if y is None:
            return y_pred
        else:
            return self.loss(y_pred, y.argmax(dim=1))


# 验证
def evaluate(model):
    correct = 0
    wrong = 0
    model.eval()
    with torch.no_grad():
        for x, y in zip(X_valid, Y_valid):
            y_pred = model(x.unsqueeze(0))
            if torch.argmax(y_pred) == torch.argmax(y):
                correct += 1
            else:
                wrong += 1
    print("验证集正确个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 预测
def predict(model_path):
    model = fully_connected(input_size=input_size, layer1_size=layer1_size, layer2_size=layer2_size,
                            output_size=output_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    correct = 0
    wrong = 0
    model.eval()
    with torch.no_grad():
        for x, y in zip(X_test, Y_test):
            y_pred = model(x.unsqueeze(0))
            if torch.argmax(y_pred) == torch.argmax(y):
                correct += 1
            else:
                wrong += 1
    print("测试集正确个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 模型训练
def main():
    model = fully_connected(input_size=input_size, layer1_size=layer1_size, layer2_size=layer2_size,
                            output_size=output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # train_x, train_y = dataset_generator(total_num)
    acc = []
    loss_epoch = []
    for epoch in range(epoch_num):
        model.train()
        loss_batch = []
        for index in range(train_num // batch_size):
            x_train = X_train[index * batch_size:(index + 1) * batch_size]
            y_train = Y_train[index * batch_size:(index + 1) * batch_size]
            loss = model(x_train, y_train)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_batch.append(loss.item())
        print("EPOCH:{}, LOSS:{:.4f}".format(epoch + 1, np.mean(loss_batch)))
        acc.append(evaluate(model))
        loss_epoch.append(np.mean(loss_batch))
    torch.save(model.state_dict(), "model_zuoye.bin")
    plt.plot(range(len(loss_epoch)), loss_epoch, label='Loss')
    plt.plot(range(len(acc)), acc, label='Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
    # print("-------------------")
    # predict("model_zuoye.bin")
