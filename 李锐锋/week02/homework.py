import torch

import torch.nn as nn
from torch.utils.data import Dataset


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.model = nn.Sequential(nn.Linear(100, 60), nn.Sigmoid(),
                                   nn.Linear(60, 5))

    def forward(self, x):
        return self.model(x)


class SimpleDataset(Dataset):
    def __init__(self, data, label):
        super(SimpleDataset, self).__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


def train(model, device, train_loader, optimizer, epochs, loss_fn):
    model.train()
    for _ in range(epochs):
        sum_loss = 0
        sum = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            target = target.squeeze()
            sum += data.size(0)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            sum_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("loss: ", sum_loss / sum)


def test(model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    sum = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target.squeeze()
            output = model(data)
            loss = loss_fn(output, target)
            test_loss += loss.item()
            sum += data.size(0)
            y_pred = output.argmax(dim=1, keepdim=False).detach().cpu()
            correct += y_pred.eq(target.to("cpu")).sum().item()
        print("accuracy: ", correct / sum)
        print("loss: ", test_loss / sum)


def create_dataset(num_train=1000, num_test=100, shape=100):
    x_train = torch.rand([num_train, shape]).float()
    x_test = torch.rand([num_test, shape]).float()
    y_train = torch.randint(0, 5, (num_train, 1))
    y_test = torch.randint(0, 5, (num_test, 1))
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_train, x_test, y_train, y_test = create_dataset()
    train_dataset = SimpleDataset(x_train, y_train)
    test_dataset = SimpleDataset(x_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 100
    train(model, device, train_loader, optimizer, num_epochs, loss_fn)
    test(model, device, test_loader, loss_fn)
