# loader.py
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def prepare_data(train_path, val_path, max_len, batch_size):
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)

    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_data['label'])
    val_labels = label_encoder.transform(val_data['label'])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = TextDataset(train_data['review'].tolist(), train_labels, tokenizer, max_len)
    val_dataset = TextDataset(val_data['review'].tolist(), val_labels, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, len(label_encoder.classes_)
