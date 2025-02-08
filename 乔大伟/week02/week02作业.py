import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 设置随机种子以便结果可复现
np.random.seed(0)
torch.manual_seed(0)

# 生成随机数据
num_samples = 1000
data = np.random.rand(num_samples, 5)
labels = np.argmax(data, axis=1)

# 将数据转换为 PyTorch 的张量
data_tensor = torch.from_numpy(data).float()
labels_tensor = torch.from_numpy(labels).long()

# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        return self.fc(x)

model = Net()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(data_tensor)
    loss = criterion(outputs, labels_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# 在测试数据上进行预测
test_data = np.random.rand(10, 5)
test_data_tensor = torch.from_numpy(test_data).float()
predictions = model(test_data_tensor)
predicted_labels = torch.argmax(predictions, dim=1).numpy()

print("预测数据:", predicted_labels)
