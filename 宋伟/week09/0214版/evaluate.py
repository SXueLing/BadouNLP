from sklearn.metrics import classification_report
import torch
import numpy as np

def evaluate(model, data_loader, device, tag2id):
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=2)
            
            for i in range(len(preds)):
                true_labels.extend([tag2id.get(t, tag2id['O']) for t in labels[i].cpu().numpy()])
                predictions.extend([tag2id.get(t, tag2id['O']) for t in preds[i].cpu().numpy()])
                
    print(classification_report(true_labels, predictions, digits=4))
