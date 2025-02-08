# coding:utf8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，找到最大的元素的下标，采用交叉熵  

"""

# 设置随机种子以确保结果可复现
torch.manual_seed(99)

# 生成数据
def generate_data(num_samples=1000):
    data = torch.randn(num_samples, 5)  # 生成 num_samples 个五维随机向量
    labels = torch.argmax(data, dim=1)  # 找到每个向量中最大值所在的维度
    return data, labels

# 定义模型
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(5, 16)  # 输入层到隐藏层
        self.fc2 = nn.Linear(16, 5)  # 隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 生成训练数据
train_data, train_labels = generate_data(num_samples=1000)
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 生成测试数据
test_data, test_labels = generate_data(num_samples=200)
test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 初始化模型、损失函数和优化器
model = SimpleClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"训练轮次 {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"正确率: {100 * correct / total}%")
