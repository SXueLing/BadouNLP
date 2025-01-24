import numpy
import torch
import torch.nn as nn
import matplotlib.pyplot as pyplot

def build_sample():
    x = numpy.random.random(5)
    return x, numpy.argmax(x)

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
    test_sample_num = 200
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if numpy.argmax(y_p) == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数%d, 正确率%f"%(correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)
        self.loss = nn.CrossEntropyLoss()
    def forward(self, x, y = None):
        y_pred = self.linear(x)
        if y is None:
            return y_pred
        else:
            return self.loss(y_pred, y)

def main():
    batch_size = 20
    epoch_num = 100
    train_sample_num = 5000
    input_size = 5
    learning_rate = 0.001

    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), learning_rate)
    log = []
    train_x, train_y = build_dataset(train_sample_num)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample_num // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, numpy.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(numpy.mean(watch_loss))])
    torch.save(model.state_dict(), "model.bin")
    print(log)
    pyplot.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    pyplot.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    pyplot.legend()
    pyplot.show()
    return

if __name__ == "__main__":
    main()
