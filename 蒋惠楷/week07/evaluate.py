# evaluate.py

from sklearn.metrics import classification_report
import torch

# 定义模型评估函数
def evaluate(model, test_loader, device, model_type):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            if model_type == "Bert":
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

            elif model_type == "XLNet":
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask)
                _, logits = outputs
                preds = torch.argmax(logits, dim=1)

            elif model_type == "LSTM" or model_type == "CNN":
                input_ids, labels = [b.to(device) for b in batch]
                outputs = model(input_ids)
                preds = torch.argmax(outputs, dim=1)
                
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")
            
            # 收集所有的预测和真实标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return classification_report(all_labels, all_preds)


