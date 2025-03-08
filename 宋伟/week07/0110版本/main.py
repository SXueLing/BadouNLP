# main.py
import torch
from config import *
from loader import prepare_data
from model import LSTMClassifier, CNNClassifier, RNNClassifier
from evaluate import evaluate_model

def train_model(model, train_loader, val_loader, device, epochs, lr):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        evaluate_model(model, val_loader, device)

def main():
    train_loader, val_loader, num_classes = prepare_data(
        TRAIN_DATA_PATH, VAL_DATA_PATH, MAX_LEN, BATCH_SIZE
    )

    models = {
        "LSTM": LSTMClassifier(vocab_size=30522, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_classes=num_classes),
        "CNN": CNNClassifier(vocab_size=30522, embedding_dim=EMBEDDING_DIM, num_classes=num_classes),
        "RNN": RNNClassifier(vocab_size=30522, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_classes=num_classes),
    }

    for model_name, model in models.items():
        print(f"正在训练模型: {model_name}")
        train_model(model, train_loader, val_loader, DEVICE, EPOCHS, LEARNING_RATE)

if __name__ == "__main__":
    main()
