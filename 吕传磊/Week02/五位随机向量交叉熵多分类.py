import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

'''
五维向量交叉熵多分类
规律：x是一个五维(索引)向量，对x做五分类任务
改用交叉熵实现一个多分类任务，五维 随机向量最大的数字在哪维就属于哪一类。
'''


# 1. 设计模型
class Work1_Torch_Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Work1_Torch_Model, self).__init__()
        # 线性层定义
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        # 线性层
        x = self.fc(x)
        return x


# 2. 构建数据集
def build_dataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        # 生成一个五维样本
        x = np.random.random(5)
        # 找到最大值所在的维度（索引）
        y = np.argmax(x)
        X.append(x)
        Y.append(y)
    X_array = np.array(X)
    Y_array = np.array(Y)
    return torch.FloatTensor(X_array), torch.LongTensor(Y_array)


# 3. 测试模型
def evaluate(model, test_x, test_y):
    model.eval()
    with torch.no_grad():
        outputs = model(test_x)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == test_y).sum().item()
        accuracy = correct / test_y.size(0)
        print(f"正确预测个数：{correct}, 正确率：{accuracy:.6f}")
        return accuracy

# 4. 训练过程
def main():
    # 训练参数
    epoch_num = 40
    batch_size = 40
    train_sample = 4000
    input_size = 5
    num_classes = 5
    learning_rate = 0.01

    # 创建训练集和测试集
    train_x, train_y = build_dataset(train_sample)
    test_x, test_y = build_dataset(1000)

    # 建立模型
    model = Work1_Torch_Model(input_size, num_classes)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 数据加载器
    # TensorDataset是一个简单的数据集类，它将输入的张量和目标张量打包在一起。你可以通过索引访问数据集中的样本。
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    # DataLoader是一个用于批量加载数据的工具。它可以从数据集中自动生成批次数据，并支持数据的随机打乱(shuffle)、多线程加载等功能。
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 开始训练
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_x, batch_y in train_loader:
            # 前向传播
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            watch_loss.append(loss.item())
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 计算平均损失
        avg_loss = np.mean(watch_loss)
        # 评估模型
        acc = evaluate(model, test_x, test_y)
        log.append([acc, avg_loss])
        print(f"Epoch [{epoch + 1}/{epoch_num}], Loss: {avg_loss:.4f}, Accuracy: {acc:.6f}")

    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 绘制损失和准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(epoch_num), [l[1] for l in log], label="Loss")
    plt.plot(range(epoch_num), [l[0] for l in log], label="Accuracy")
    plt.legend()
    plt.xlabel("Epoch")
    plt.show()


if __name__ == "__main__":
    main()
