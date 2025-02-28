import torch
import torch.nn as nn
import torch.optim as optim
import random
import string

# 定义超参数
vocab_size = len(string.ascii_lowercase) + 1  # 包含一个特殊字符
embedding_dim = 16
hidden_dim = 32
# 修改 num_classes 为 11 以覆盖标签 0 到 10
num_classes = 11
num_epochs = 100
learning_rate = 0.01

# 数据生成函数
def generate_data(num_samples):
    data = []
    for _ in range(num_samples):
        length = random.randint(1, num_classes - 1)  # 最大长度为 num_classes - 1
        # 随机生成字符串
        s = ''.join(random.choices(string.ascii_lowercase, k=length))
        # 随机选择一个特定字符
        target_char = random.choice(string.ascii_lowercase)
        # 插入特定字符
        position = random.randint(0, length)
        s = s[:position] + target_char + s[position:]
        # 生成标签
        label = position
        data.append((s, label))
    return data

# 字符到索引的映射
char_to_idx = {char: idx + 1 for idx, char in enumerate(string.ascii_lowercase)}
char_to_idx['<PAD>'] = 0

# 数据转换函数
def convert_to_tensor(s):
    return torch.tensor([char_to_idx[char] for char in s], dtype=torch.long)

# 构建 RNN 模型
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.rnn(embedded)
        out = self.fc(out[:, -1, :])
        return out

# 生成数据集
train_data = generate_data(1000)
test_data = generate_data(200)

# 初始化模型、损失函数和优化器
model = RNN(vocab_size, embedding_dim, hidden_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    total_loss = 0
    for s, label in train_data:
        input_tensor = convert_to_tensor(s).unsqueeze(0)
        label_tensor = torch.tensor([label], dtype=torch.long)

        # 前向传播
        output = model(input_tensor)
        loss = criterion(output, label_tensor)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_data)}')

# 评估模型
correct = 0
total = 0
with torch.no_grad():
    for s, label in test_data:
        input_tensor = convert_to_tensor(s).unsqueeze(0)
        label_tensor = torch.tensor([label], dtype=torch.long)

        output = model(input_tensor)
        _, predicted = torch.max(output.data, 1)
        total += 1
        correct += (predicted == label_tensor).sum().item()

print(f'Accuracy on test set: {100 * correct / total}%')
