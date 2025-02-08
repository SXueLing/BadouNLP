import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchsummary import summary


def compute_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    target_class = target.argmax(dim=1)
    correct = (predicted == target_class).sum().item()
    accuracy = correct / target.size(0)
    return accuracy


class Classifier(nn.Module):
    def __init__(self, num_classes, num_input_features, num_layers, device):
        super().__init__()

        self.device = device

        layers = []
        last_dim = num_input_features
        for i in range(num_layers - 1):
            layers.append(nn.Linear(last_dim, last_dim * 2))
            layers.append(nn.ReLU())
            last_dim *= 2
        layers.append(nn.Linear(last_dim, num_classes))

        self.model = nn.Sequential(*layers).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)

        # Plot data
        self.losses = []
        self.accuracies = []

    def forward(self, x):
        return self.model(x)

    def train_model(self, dataloader, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            for i, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                loss = loss.item()
                self.losses.append(loss)
                accuracy = compute_accuracy(output, target)
                self.accuracies.append(accuracy)
                print(f"Epoch [{epoch + 1}/{num_epochs}] Batch {i}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    def plot(self):
        plt.figure(figsize=(12, 5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.losses, label='Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.accuracies, label='Accuracy', color='orange')
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


class MyDataset(Dataset):
    def __init__(self, num_data, num_classes, device):
        self._x = torch.randn(num_data, num_classes, device=device)

        # Get the index of the maximum value in each row to simulate class labels
        class_indices = self._x.argmax(dim=1, keepdim=True)  # Shape: (num_data, 1)

        # One-hot encode the class indices
        self._y = torch.zeros(num_data, num_classes, device=device)  # Shape: (num_data, num_classes)
        self._y.scatter_(1, class_indices, 1)

    def __len__(self):
        return len(self._x)

    def __getitem__(self, idx):
        return self._x[idx], self._y[idx]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    classifier = Classifier(5, 5, 2, device)
    dataset = MyDataset(10000, 5, device=device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    sample_x, _ = dataset[0]
    summary(classifier, input_size=(sample_x.shape[0],))

    classifier.train_model(dataloader, 100)
    classifier.plot()
