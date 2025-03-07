import torch
from transformers import AutoTokenizer
from config import Config
from model import LoRANERModel

def predict(text, model, tokenizer, device):
    model.eval()
    encoding = tokenizer(text, truncation=True, padding="max_length", max_length=Config.max_length, return_tensors="pt")
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = outputs.logits.argmax(dim=-1).cpu().numpy().flatten()

    return predictions

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    model = LoRANERModel(num_labels=9)  # 假设有9个实体类别
    model.load_state_dict(torch.load("best_model.pth"))  # 加载训练好的模型
    model.to(Config.device)

    test_text = "Barack Obama was the 44th President of the United States."
    preds = predict(test_text, model, tokenizer, Config.device)
    print("Predicted Labels:", preds)
