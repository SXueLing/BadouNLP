'''
实现一个自行构造的找规律(机器学习)任务
五维判断: x是一个5维向量, 向量中哪个标量最大就输出哪一维下标
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from MCrossEntropyLoss import MCrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 设置随机种子使结果可复现
np.random.seed(42)
torch.manual_seed(42)

# 1. 数据生成：五位随机向量和对应的标签
def generate_data(batch_size):
    # 生成五位随机向量
    X = np.random.rand(batch_size, 5).astype(np.float32)
    # 获取每个样本最大值所在的维度
    Y = np.argmax(X, axis=1)
    return torch.tensor(X), torch.tensor(Y)

# 2. 模型建立
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 5)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

def main():
    # 3. 模型实例及定义损失函数和优化器
    model = SimpleNN()
    # criterion = nn.CrossEntropyLoss() # pytorch内置交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Adam优化器

    # 4. 生成训练数据
    X_train, Y_train = generate_data(2000)

    # 5. 数据加载
    train_data = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # 用来存储每个 epoch 的 loss 和 accuracy
    all_losses = []
    all_accuracies = []

    # 6. 模型训练及保存
    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad() # 清除上一步的梯度
            outputs = model(inputs)
            loss = MCrossEntropyLoss(outputs, labels) # 手动实现交叉熵损失
            loss.backward() # 反向传播
            optimizer.step() # 更新参数

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 计算当前 epoch 的平均损失和准确率
        train_accuracy = correct / total
        all_losses.append(running_loss / len(train_loader))  # 每个 epoch 的平均损失
        all_accuracies.append(train_accuracy)  # 每个 epoch 的训练准确率

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}')
    # 保存模型
    torch.save(model.state_dict(), './simple_nn_model.pt')

    # 绘制损失和准确率图像
    plot_metrics(all_losses, all_accuracies)

    return

def plot_metrics(losses, accuracies):
    """
    绘制损失和准确率变化图
    """
    epochs = range(1, len(losses) + 1)

    # 创建一个新的图形
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    # 显示图形
    plt.tight_layout()
    plt.show()


# 7. 模型测试
def predict(model_path, test_loader):
    model = SimpleNN()
    model.load_state_dict(torch.load(model_path))  # 加载训练好的模型

    model.eval()  # 切换到评估模式
    # 初始化准确率计算
    correct = 0
    total = 0

    with torch.no_grad():  # 不计算梯度
        for inputs, labels in test_loader:
            outputs = model(inputs)  # 模型预测
            _, predicted = torch.max(outputs, 1)  # 获取最大概率的类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total  # 计算准确率
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
    # 测试数据
    X_test, Y_test = generate_data(400)
    test_data = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    predict("./simple_nn_model.pt", test_loader)
