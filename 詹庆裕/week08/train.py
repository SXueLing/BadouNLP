from evaluate import model_evaluate
from config import Config
from model import Model, choose_optimizer
from loader_train_data import MyDataLoad
import torch
import numpy as np


if __name__ == "__main__":
    data = MyDataLoad("data/data.json")
    model = Model()
    optimizer = choose_optimizer(model)
    seq_encode = []
    for epoch in range(Config["epoch"]):
        loss_item = []
        for batch in data.dataloader:
            if Config["loss_type"] == "triplet_loss":
                x1, x2, label = batch["anchor"],batch["pos"],batch["neg"]
            else:
                x1, x2, label = batch["input_ids1"],batch["input_ids2"],batch["labels"]
            optimizer.zero_grad()
            loss = model(x1, x2, label)
            loss.backward()
            optimizer.step()
            loss_item.append(loss.item())
        avg_loss = sum(loss_item) / len(loss_item)
        print("第%d轮的loss为%f"%(epoch+1, avg_loss))
        seq_encode = model_evaluate(model, data.dataset)
    path = f"model_weight/{Config["model_type"]}_model_{Config["loss_type"]}.pth"
    torch.save(model.state_dict(), path)
    seq_encode_path = f"seq_encode/{Config["loss_type"]}_seq_encode.npy"
    np.save(seq_encode_path, seq_encode)







