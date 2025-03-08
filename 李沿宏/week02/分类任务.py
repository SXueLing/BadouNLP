import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义模型
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # 全连接层

    def forward(self, x):
        return self.fc(x)  # 输出logits

# 生成随机数据
def generate_random_data(batch_size, input_dim):
    data = torch.rand(batch_size, input_dim)  # 生成随机向量
    labels = torch.argmax(data, dim=1)  # 最大值的索引作为标签
    return data, labels

# 参数设置
input_dim = 5  # 输入维度（五维向量）
output_dim = 5  # 输出维度（五类）
batch_size = 32
learning_rate = 0.05
num_epochs = 600

# 初始化模型、损失函数和优化器
model = SimpleClassifier(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    # 生成随机数据
    inputs, labels = generate_random_data(batch_size, input_dim)

    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印训练信息
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
def test_model(model, test_size=10):
    model.eval()  # 切换到评估模式
    with torch.no_grad():
        test_data, test_labels = generate_random_data(test_size, input_dim)
        predictions = torch.argmax(model(test_data), dim=1)
        accuracy = (predictions == test_labels).float().mean()
        print(f'Test Accuracy: {accuracy.item() * 100:.2f}%')

# 测试模型性能
test_model(model)
