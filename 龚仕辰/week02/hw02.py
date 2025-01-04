import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from torch.utils.data import TensorDataset, DataLoader
def generate_data(num_samples=1000, num_classes=5):
    X = np.random.rand(num_samples, num_classes)
    y = np.argmax(X, axis=1)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

class SimpleNet(nn.Module):
    def __init__(self, input_layer, hidden_layer, output_layer, batch_size=32):
        super(SimpleNet, self).__init__()
        self.batch_size = batch_size
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.fc1 = nn.Linear(input_layer, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, output_layer)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

criterion = nn.CrossEntropyLoss()

def train_model(model, X, y, epochs=100, batch_size=32):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= len(dataloader.dataset)

        if (epoch + 1) % batch_size == 0:
            print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')

def evaluate_model(model, X, y):
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)
        total = y.size(0)
        correct = (predicted == y).sum().item()
        print(f'Accuracy: {100 * correct / total}%')
        return predicted.numpy()  # 确保返回预测结果

# Step 6: Visualize results
def visualize_results(X, y, predicted):
    # 使用 PCA 将特征降维到 2D 以便可视化
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 6))
    
    # 绘制真实标签的数据点
    for i in range(5):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=f'True Class {i}', alpha=0.6)
    
    # 绘制预测标签的数据点
    for i in range(5):
        plt.scatter(X_pca[predicted == i, 0], X_pca[predicted == i, 1], marker='x', label=f'Predicted Class {i}', alpha=0.6)
    
    plt.title('True vs Predicted Classes')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()

# Main
X, y = generate_data()
model = SimpleNet(5, 10, 5, 64)
train_model(model, X, y, epochs=2048, batch_size=model.batch_size)
predicted = evaluate_model(model, X, y)
visualize_results(X.numpy(), y.numpy(), predicted)
