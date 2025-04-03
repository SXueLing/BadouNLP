import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from loader import get_dataloaders
from model import LoRANERModel
from evaluation import evaluate

def train():
    train_loader, val_loader = get_dataloaders()
    model = LoRANERModel(num_labels=9).to(Config.device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = 0.0
    for epoch in range(Config.num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            labels = batch["labels"].to(Config.device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

        # 评估
        report = evaluate(model, val_loader, Config.device)
        print(report)

        # 保存最佳模型
        if "f1-score" in report and report["f1-score"] > best_f1:
            best_f1 = report["f1-score"]
            torch.save(model.state_dict(), "best_model.pth")

if __name__ == "__main__":
    train()
