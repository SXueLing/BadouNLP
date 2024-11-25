import torch
import torch.nn as nn
import numpy as np


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)

    def forward(self, x):
        x = self.linear(x)
        y = nn.functional.softmax(x, dim=1)
        return y


def build_sample():
    x = np.random.random(5)
    y = [0, 0, 0, 0, 0]
    y[np.argmax(x)] = 1
    return x, y


def build_dataset(n_sample):
    X, Y = [], []
    for i in range(n_sample):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(Y))


def evaluate(model, n_sample):
    model.eval()
    x, y = build_dataset(n_sample)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for i in range(n_sample):
            if np.argmax(y[i]) == np.argmax(y_pred[i]):
                correct += 1
            else:
                wrong += 1
    acc = correct / (correct + wrong)
    # print("Correct: %d, Acc: %f" % (correct, acc))
    return acc


def train():
    n_epoch = 100
    batch_size = 20
    n_train_sample = 5000
    input_size = 5
    lr = 0.001

    train_x, train_y = build_dataset(n_train_sample)

    model = TorchModel(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epoch):
        model.train()
        epoch_loss = []
        for batch_idx in range(n_train_sample // batch_size):
            x = train_x[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            y = train_y[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss.append(loss.item())
        test_acc = evaluate(model, 100)
        print("Epoch: %d, avg loss: %f, test acc: %f" % (epoch+1, float(np.mean(epoch_loss)), float(test_acc)))
    return model


def predict(model_path, test_data):
    model = TorchModel(5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(test_data))
    for vec, res in zip(test_data, result):
        print("Input:", vec, "Output:", res, "Classify:", np.argmax(res))


if __name__ == "__main__":
    model = train()
    # torch.save(model.state_dict(), "TorchDemo.pt")

    # test_data = [[0.4839, 0.3915, 0.9087, 0.4887, 0.8435],
    #     [0.1167, 0.6241, 0.5353, 0.3902, 0.3305],
    #     [0.5342, 0.1803, 0.2009, 0.0835, 0.3376],
    #     [0.4152, 0.2863, 0.3440, 0.6784, 0.3948],
    #     [0.6662, 0.1946, 0.8714, 0.7610, 0.5002]]
    # predict("TorchDemo.pt", test_data)
