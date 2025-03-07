import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class NERDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts, self.labels = self.load_data(data_path)

    def load_data(self, data_path):
        texts = []
        labels = []
        with open(data_path, 'r') as f:
            sentence = []
            label = []
            for line in f:
                if line.strip():
                    word, tag = line.split()
                    sentence.append(word)
                    label.append(tag)
                else:
                    texts.append(' '.join(sentence))
                    labels.append(label)
                    sentence = []
                    label = []
        return texts, labels

    def encode_tags(self, labels, tag2id):
        return [tag2id.get(tag, tag2id['O']) for tag in labels]

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        label_ids = self.encode_tags(labels, tag2id)
        label_ids = label_ids[:self.max_len] + [tag2id['O']] * (self.max_len - len(label_ids))  # Padding
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label_ids)
        }

    def __len__(self):
        return len(self.texts)

def create_data_loader(data_path, tokenizer, max_len, batch_size):
    dataset = NERDataset(data_path, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size)
