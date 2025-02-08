import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 数据生成：生成样本，返回输入字符串和标签，这里的特点字符为`a`
def generate_data(target_char='a', num_samples=1000, string_length=4):
    X = []
    y = []
    for _ in range(num_samples):
        # 随机生成一个长度为string_length的字符串
        string = ''.join(np.random.choice(
            list('abcdefghijklmnopqrstuvwxyz'), string_length))
        X.append(string)
        # 判断目标字符的位置
        target_pos = string.find(target_char)
        if target_pos == -1:
            y.append(-1)  # 目标字符未出现，标签为-1
        else:
            y.append(target_pos)  # 目标字符第一次出现的位置
    return np.array(X), np.array(y)

# One-Hot编码
def string_to_one_hot(strings, vocab_size=26):
    def char_to_index(c):
        return ord(c) - ord('a')  # 字符转为索引（a->0, b->1, ..., z->25）

    # 将字符串拆成字符，并确保是二维数组：样本数 × 字符串长度
    strings = np.array([list(string) for string in strings])

    # 这里不使用embedding编码方式，创建空的one-hot矩阵，形状为[样本数, 字符串长度, 字母表大小]
    one_hot_encoded = np.zeros(
        (strings.shape[0], strings.shape[1], vocab_size))

    for i, string in enumerate(strings):
        for j, char in enumerate(string):
            one_hot_encoded[i, j, char_to_index(char)] = 1
    return one_hot_encoded

# 生成数据集
X_train, y_train = generate_data(target_char='a', num_samples=1000)
X_test, y_test = generate_data(target_char='a', num_samples=200)

# 将数据转换为One-Hot编码
X_train_one_hot = string_to_one_hot(X_train)
X_test_one_hot = string_to_one_hot(X_test)

# 将数据转为torch张量
X_train_tensor = torch.tensor(X_train_one_hot, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # 标签类型为long
X_test_tensor = torch.tensor(X_test_one_hot, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


class PositionPredictor(nn.Module):
    """docstring for PositionPredictor"""

    def __init__(self, input_size=26, hidden_size=64, output_size=1):
        super(PositionPredictor, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)  # 使用RNN进行特征提取
        out = self.fc(rnn_out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 创建模型
model = PositionPredictor(input_size=26, hidden_size=64, output_size=1)

# 损失函数：这里使用L1损失，适合回归任务
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
num_epoch = 20
for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())  # 回归任务使用MSE
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"第{epoch+1}/{num_epoch}批次，损失值为：{running_loss/len(train_loader)}")

# 测试
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = torch.round(outputs.squeeze())  # 对输出进行四舍五入，得到预测位置
        total += labels.size(0)
        correct += (predicted == labels).sum().item()  # 计算预测是否与实际位置相等

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


def predict(model, input_str, target_char='a'):
    model.eval()
    input_one_hot = string_to_one_hot(np.array([input_str]))  # 转换为One-Hot编码
    input_tensor = torch.tensor(input_one_hot, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor)
        predicted_pos = prediction.item()
        return int(predicted_pos) if predicted_pos != -1 else -1  # 如果预测为-1，则返回-1

# 示例预测
print(predict(model, "bcda"))  # 包含'a'，预测为2（目标字符'a'第一次出现的位置）
print(predict(model, "bcde"))  # 不包含'a'，预测为-1

# 预测准确率：Test Accuracy: 97.50%
