import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MultiClassModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiClassModel, self).__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

def build_sample():
    x = np.random.random(5)
    return x, np.argmax(x)

def generate_data(batch_size):
    X = []
    Y = []
    for _ in range(batch_size):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long)


def train_model(model, epochs, batch_size, learning_rate):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for batch in range(0, 50000, batch_size):  # 50000
            x_batch, y_batch = generate_data(batch_size)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
    torch.save(model.state_dict(), "model.bin")

def check_value(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(100):  # 测试100个样本
            x_test, y_test = generate_data(1)
            outputs = model(x_test)
            _, predicted = torch.max(outputs.data, 1)
            total += y_test.size(0)
            correct += (predicted == y_test).sum().item()  # 预测索引下标相等就为正确值
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')
    return accuracy

def main():
    model = MultiClassModel(input_size=5, num_classes=5)
    train_model(model, epochs=10, batch_size=32, learning_rate=0.001)
    check_value(model)

if __name__ == '__main__':
    main()