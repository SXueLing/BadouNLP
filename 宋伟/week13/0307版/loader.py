import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from config import Config

class NERDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.data = json.load(open(file_path, "r", encoding="utf-8"))
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        labels = self.data[idx]["labels"]  # 标签应该是 token 对齐的格式
        
        encoding = self.tokenizer(
            text, 
            truncation=True, 
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        label_ids = torch.tensor(labels + [0] * (self.max_length - len(labels)))

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_ids}

def get_dataloaders():
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    train_dataset = NERDataset(Config.train_file, tokenizer, Config.max_length)
    val_dataset = NERDataset(Config.val_file, tokenizer, Config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size)

    return train_loader, val_loader
