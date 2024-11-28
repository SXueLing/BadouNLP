"""

任务详情，请见README

"""

# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes) # Linear layer (5 -> 3)
        self.loss = nn.CrossEntropyLoss() # Cross Entropy loss

    def forward(self, x, y=None):
        logits = self.linear(x) # Linear transformation to logits
        if y is not None:
            return self.loss(logits, y) # Compute Cross Entropy loss
        else:
            return torch.softmax(logits, dim=1) # Convert logits to probabilities for inference

# generate a single sample based on the classification rule
def build_sample():
    x = np.random.random(5) # Generate random 5-dimensional vector
    sum1 = float(x[0] + x[1])  # First two components
    sum2 = float(x[2] + x[3])  # Middle two components
    sum3 = float(x[4] + x[0])  # Last and first components
    if sum1 > max(sum2, sum3):
        return x, 0 # Class 0
    if sum2 > max(sum1, sum3):
        return x, 1 # Class 1
    else:
        return x, 2 # Class 2

# generate dataset
def build_dataset(total_sample_num):
    x = []
    y = []
    for i in range(total_sample_num):
        sample, label = build_sample()
        x.append(sample)
        y.append(label)
    return torch.FloatTensor(np.array(x)), torch.LongTensor(y) # Return tensors (features, labels)

# evaluate the model
def evaluate(model):
    model.eval() # Set the model to evaluation mode
    test_sample_num = 100
    x, y = build_dataset(test_sample_num) # Generate 100 samples
    print("There are %d class 0 samples, %d class 1 samples, and %d class 2 samples in the test set" % (sum(y == 0), sum(y == 1), sum(y == 2)))
    y_pred = model(x).argmax(dim=1) # Predict the labels
    # Count correct predictions
    correct = (y_pred == y).sum().item()
    accuracy = correct / test_sample_num
    print(f"Accuracy: {accuracy:.4f} ({correct}/{test_sample_num})")
    return accuracy

# train the model
def train():
    # Hyperparameters
    epoch_num = 20 # Number of epochs
    batch_size = 20 # Batch size
    train_sample = 5000 # Number of training samples
    input_size = 5 # Input size
    num_classes = 3 # Number of classes
    learning_rate = 0.01 # Learning rate

    # initialize the model and optimizer
    model = TorchModel(input_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam Gradient Descent

    # Train the model
    log = []
    for epoch in range(epoch_num):
        model.train() # Set the model to training mode
        train_x, train_y = build_dataset(train_sample) # Generate training samples
        watch_loss = []
        for batch_index in range(0, train_sample, batch_size):
            x = train_x[batch_index: batch_index + batch_size]
            y = train_y[batch_index: batch_index + batch_size]
            loss = model(x, y) # Compute loss
            optimizer.zero_grad() # Reset gradients
            loss.backward() # Compute gradients
            optimizer.step() # Update weights
            watch_loss.append(loss.item()) # Store loss

        # Print epoch stats
        print(f"Epoch {epoch + 1}/{epoch_num}, Loss: {np.mean(watch_loss):.4f}")
        acc = evaluate(model) # Evaluate accuracy
        log.append([acc, np.mean(watch_loss)]) # Store accuracy and loss

    # Save the model
    torch.save(model.state_dict(), "multiclass_model.bin")

    # Plot the accuracy and loss curves
    plt.plot(range(len(log)), [l[0] for l in log], label="Accuracy")
    plt.plot(range(len(log)), [l[1] for l in log], label="Loss")
    plt.legend()
    plt.show()

# Predict function
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 3
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(input_vec)
        y_pred = model(x).argmax(dim=1) # Predict the label
    for vec, res in zip(input_vec, y_pred):
        predicted_class = res.item()
        print(f"Input: {vec}, Predicted Class: {predicted_class}, Probabilities: {res.numpy()}")

if __name__ == "__main__":
    train()

