import torch
from sklearn.metrics import classification_report

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.argmax(dim=-1)

            all_preds.extend(logits.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    return classification_report(all_labels, all_preds, zero_division=0)
