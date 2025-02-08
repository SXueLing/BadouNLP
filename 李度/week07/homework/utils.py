import torch
import random
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer

random.seed(42)


def load_data(
    data_path,
    tokenizer,
    num_worker=2,
    train_size=0.7,
    val_size=0.3,
    batch_size=64,
    max_length=256,
):
    df = pd.read_csv(data_path)
    raw_data = np.array(df)
    pos_data = raw_data[raw_data[:, 0] == 1]
    neg_data = raw_data[raw_data[:, 0] != 1]
    pos_train_data, pos_val_data = train_test_split(pos_data, train_size=train_size, test_size=val_size)
    neg_train_data, neg_val_data = train_test_split(neg_data, train_size=train_size, test_size=val_size)
    pos_train_review, pos_train_label = pos_train_data[:, 1], pos_train_data[:, 0]
    neg_train_review, neg_train_label = neg_train_data[:, 1], neg_train_data[:, 0]
    pos_val_review, pos_val_label = pos_val_data[:, 1], pos_val_data[:, 0]
    neg_val_review, neg_val_label = neg_val_data[:, 1], neg_val_data[:, 0]
    train_data = np.concatenate((pos_train_review, neg_train_review), axis=0)
    train_label = np.concatenate((pos_train_label, neg_train_label), axis=0)
    val_data = np.concatenate((pos_val_review, neg_val_review), axis=0)
    val_label = np.concatenate((pos_val_label, neg_val_label), axis=0)
    train_loader = DataLoader(
        Dataset(train_data, train_label, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_worker,
    )
    val_loader = DataLoader(
        Dataset(val_data, val_label, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, val_loader


def load_tokenizer(tokenizer):
    if not tokenizer:
        tokenizer = BertTokenizer.from_pretrained(
            "/Users/duli/Desktop/UT Austin/自学内容/八斗课件/NLP/six-lm/bert-base-chinese")
    else:
        tokenizer = tokenizer.from_pretrained(
            "/Users/duli/Desktop/UT Austin/自学内容/八斗课件/NLP/six-lm/bert-base-chinese")
    return tokenizer


class Dataset:
    def __init__(self, data, label, tokenizer, max_length):
        self.data = data
        self.label = torch.LongTensor(label.astype(np.uint8))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.id_inputs = self._char_to_id()

    def _char_to_id(self):
        result = []
        for sentence in self.data:
            result.append(
                self.tokenizer.encode(
                    sentence,
                    max_length=self.max_length,
                    truncation=True,
                    padding='max_length'
                )
            )
        return torch.tensor(result, dtype=torch.long)

    def __len__(self):
        return len(self.id_inputs)

    def __getitem__(self, item):
        return self.id_inputs[item], self.label[item]
