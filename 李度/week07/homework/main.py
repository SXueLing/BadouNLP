import json
import os
from datetime import datetime

import torch
from torchmetrics import Accuracy

from models import TorchModel
from config import load_config
from utils import load_data, load_tokenizer


def train():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    cfg = load_config()
    tokenizer = load_tokenizer(cfg["tokenizer"] if "tokenizer" in cfg else None)
    model = TorchModel(
        vocab=tokenizer.vocab,
        core_model=cfg["model"],
        pooling=cfg["pooling"],
        num_layers=cfg["encoder_layers"],
        use_bert_pooler=cfg["use_bert_pooler"],
    )
    model = model.to(device)
    train_loader, val_loader = load_data(
        "../文本分类练习.csv",
        tokenizer=tokenizer,
        num_worker=os.cpu_count(),
        train_size=0.7,
        val_size=0.3,
        batch_size=64,
        max_length=cfg["seq_max_length"],
    )
    optimizer = cfg["optimizer"](model.parameters(), lr=cfg["learning_rate"])
    loss_fn = cfg["loss"]()
    num_epoch = cfg["epochs"]
    train_metrics, val_metrics = (Accuracy(task="multiclass", num_classes=2).to(device),
                                  Accuracy(task="multiclass", num_classes=2).to(device))
    result = []
    best_acc = 0
    best_state = None
    for i in range(num_epoch):
        model.train()
        train_loss, val_loss = 0, 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            train_metrics.update(y_pred, y)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                val_metrics.update(y_pred.to(device), y)
                loss = loss_fn(y_pred, y)
                val_loss += loss.item()

        train_acc = train_metrics.compute()
        val_acc = val_metrics.compute()
        train_metrics.reset()
        val_metrics.reset()
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()

        print(f"Epoch {i + 1}/{num_epoch}, "
              f"Train Total Loss: {train_loss:.4f}, \n"
              f"Train Avg Loss: {train_loss / len(train_loader):.4f}, \n"
              f"Val Loss: {val_loss:.4f}, \n"
              f"Val Avg Loss {val_loss / len(val_loader):.4f} \n "
              f"Train Acc: {train_acc:.4f}, \n"
              f"Validation Acc: {val_acc:.4f}")
        run_log = {
            "Epoch": num_epoch,
            "Train Total Loss": train_loss,
            "Train Avg Loss": train_loss / len(train_loader),
            "Val Loss": val_loss,
            "Val Avg Loss": val_loss / len(val_loader),
            "Train Acc": train_acc,
            "Validation Acc": val_acc,
        }
        result.append(run_log)

    # save the result, and file name is model_name+timestamp format
    model_name = cfg["model"]  # replace with your model name
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    dir_path = f"./log/{model_name}_{timestamp}"
    os.makedirs(dir_path, exist_ok=True)
    save_path = os.path.join(dir_path, "result.json")
    with open(save_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Model trained and runtime log saved to {save_path}")
    torch.save(best_state, f"./log/model.pth")


if __name__ == "__main__":
    train()
