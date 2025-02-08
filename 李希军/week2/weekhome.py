import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 数据生成
def generate_data(num_samples=1000):
    data = torch.randn(num_samples, 5)  # 生成五维随机向量
    labels = torch.argmax(data, dim=1)  # 最大值的索引作为类别标签
    return data, labels

# 模型定义
class MultiClassModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MultiClassModel, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)  # 全连接层
    
    def forward(self, x):
        return self.fc(x)  # 输出 logits

# 评估函数
def evaluate(model, criterion, data, labels):
    model.eval()
    with torch.no_grad():
        logits = model(data)  # 模型预测
        loss = criterion(logits, labels)  # 计算损失
        preds = torch.argmax(logits, dim=1)  # 获取预测的类别
        accuracy = (preds == labels).float().mean()  # 计算准确率
    return loss.item(), accuracy.item()

# 主函数
def main():
    # 参数配置
    num_classes = 5
    input_dim = 5
    num_samples = 5000
    batch_size = 32
    learning_rate = 0.01
    epochs = 20

    # 数据准备
    train_data, train_labels = generate_data(num_samples)
    test_data, test_labels = generate_data(500)

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 模型、损失函数和优化器
    model = MultiClassModel(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练过程
    train_losses = []
    test_accuracies = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = model(batch_data)  # 前向传播
            loss = criterion(logits, batch_labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        # 测试模型
        test_loss, test_accuracy = evaluate(model, criterion, test_data, test_labels)
        test_accuracies.append(test_accuracy)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss / len(train_loader):.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # 绘制损失和准确率曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()

    return model

if __name__ == "__main__":
    model = main()
