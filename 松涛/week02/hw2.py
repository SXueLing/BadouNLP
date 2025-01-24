import torch
import torch.nn as nn
import numpy as np

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        y_pred = self.linear(x)  # 保持 logits
        if y is not None:
            y = y.long()  # 转换为 (batch_size,) 整数类型
            return self.loss(y_pred, y)
        else:
            return torch.argmax(y_pred, dim=1)  # 返回预测类别

def build_sample():
    x = np.random.random(5)
    return x, np.argmax(x)

def build_dataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
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
            if y_p == y_t:
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
    learning_rate = 0.01

    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    train_x, train_y = build_dataset(train_sample)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):    
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), "model.bin")
    print(log)

def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果
        
        
if __name__ == "__main__":
    # main()
    test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.bin", test_vec)