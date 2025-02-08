import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)
        # self.activation = torch.sigmoid
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        logits = self.linear(x)
        if y is not None:
            return self.loss(logits, y)
        else:
            return logits
        
def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)
    return x, max_index
    # if x[0] > x[4]:
    #     return x, 1
    # else:
    #     return x, 0

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    # correct, wrong = 0, 0
    correct = 0

    with torch.no_grad():
        logits = model(x)
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == y).sum().item()

    accuracy = correct / test_sample_num
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy
        # y_pred = model(x)
        # for y_p, y_t in zip(y_pred, y):
        #     if float(y_p) < 0.5 and int(y_t) == 0:
        #         correct += 1
        #     elif float(y_p) >= 0.5 and int(y_t) == 1:
        #         correct += 1
        #     else:
        #         wrong += 1
    # print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    # if wrong == 0:
    #     return 1.0 if correct > 0 else 0.0  # 如果 wrong 为零，返回 1.0 或 0.0
    # else:
    #     return correct / (correct + wrong)

def main():
    epoch_num = 1000
    batch_size = 40
    train_sample = 10000
    input_size = 5
    learning_rate = 0.005
    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)
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
            
        print("======\n average loss for  %d round: %f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    torch.save(model.state_dict(), "model.bin")

    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    return

def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(input_vec))
        probabilities = torch.softmax(logits, dim = 1)
        predictions = torch.argmax(probabilities, dim = 1)

    for vec, prob, pred in zip(input_vec, probabilities, predictions):
        print(f"Input: {vec}, Predicted Categpry: {pred.item()}, Probabilities: {prob.numpy()}")
    # with torch.no_grad():
    #     result = model.forward(torch.FloatTensor(input_vec))
    # for vec, res in zip(input_vec, result):
    #     print("input: %s, prediction category: %d, probability: %f" % (vec, round(float(res), res)))

if __name__ == "__main__":
    main()
