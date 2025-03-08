import torch
import torch.nn as nn
import torch.optim as optim

# 定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 生成随机输入数据和对应的标签
# 假设batch_size为10，每个样本是一个5维的向量
batch_size = 10
input_data = torch.randn(batch_size, 5)
# 找到每个样本中最大值的索引作为标签
labels = torch.argmax(input_data, dim=1)

# 定义一个简单的模型，这里只是一个线性层，因为输入已经是特征向量
model = nn.Linear(5, 5)

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型（这里简单迭代几次展示过程）
for epoch in range(5):
    # 前向传播
    outputs = model(input_data)
    loss = criterion(outputs, labels)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# 测试模型，对新的随机数据进行预测
test_data = torch.randn(batch_size, 5)
with torch.no_grad():
    test_outputs = model(test_data)
    predicted = torch.argmax(test_outputs, dim=1)

print("预测结果:", predicted)
