import torch
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


X = torch.randn(100, 5)
Y = X.argmax(dim=-1)

epochs = 100
batch_size = 5
cross_entropy_loss = nn.CrossEntropyLoss()
model = Net(5, 128, 5)
optim = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(epochs):
    loss_sum = 0
    for i in range(0, X.shape[0], batch_size):
        x_batch = X[i:i + batch_size]
        y_batch = Y[i:i + batch_size]
        pred = model(x_batch)
        loss = cross_entropy_loss(pred, y_batch)
        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_sum += loss.item() * batch_size
    print(f'Epoch {epoch + 1}, Loss: {loss_sum / X.shape[0]}')

X = [
    [0.1, 0.2, 12, 0.4, 0.5],
    [34, 11, 50, 21, 22],
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [12, 34, 56, 21, 34],
    [34, 21, 12, 35, 21],
    [456, 12, 34, 21, 34]
]
X = torch.tensor(X, dtype=torch.float32)
Y = X.argmax(dim=-1)
pred = model(X)
print(pred.argmax(dim=-1))
print(f"acc: {(pred.argmax(dim=-1) == Y).sum().item() / X.shape[0]}")
