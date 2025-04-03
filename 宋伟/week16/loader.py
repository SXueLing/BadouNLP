import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class DialogueDataset(Dataset):
    def __init__(self, dialogues, tokenizer, max_len):
        self.dialogues = dialogues
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, item):
        dialogue = self.dialogues[item]
        input_ids = self.tokenizer.encode(dialogue, truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt").squeeze(0)
        return {"input_ids": input_ids}

def load_data(file_path, tokenizer, max_len, batch_size):
    with open(file_path, 'r') as f:
        dialogues = f.readlines()
    
    dataset = DialogueDataset(dialogues, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
