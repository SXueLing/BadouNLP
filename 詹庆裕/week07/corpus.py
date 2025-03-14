import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from config import Config
import os


class CacheDataset(Dataset):
    def __init__(self, cache_path):
        self.data = torch.load(cache_path)

    def __len__(self):
        return len(self.data["labels"])

    def __getitem__(self, index):
        return {
            "input_ids": self.data["input_ides"][index],
            "attention_mask": self.data["attention_mask"][index],
            "labels": self.data["labels"][index]
        }


class OptimizedDataLoader:
    def __init__(self, path):
        self.path = path
        self.cache_path = f"{path}.pt"
        self.data = None

        if os.path.exists(self.cache_path):
            self.data = CacheDataset(self.cache_path)
        else:
            self._initialize_components()
            self._build_cache()
            self.dataset = CacheDataset(self.cache_path)

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=Config["batch_size"],
            shuffle=True,
            num_workers=Config["num_workers"],
            drop_last=True
        )

    def _initialize_components(self):
        with open(self.path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]
            self.label2id = {label: idx for idx, label in enumerate({item["label"] for item in self.data})}
        if Config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(
                Config["model_name_or_path"],
                use_fast=True)
            self.max_length = min(self.tokenizer.model_max_length, 512)
        else:
            with open(Config["vocab"], "r", encoding="utf-8"):
                self.vocab = {word.strip(): idx+1 for idx, word in enumerate(f)}

    def _build_cache(self):
        contents = [item["content"] for item in self.data]
        labels = [self.label2id[item["label"]] for item in self.data]
        if Config["model_type"] == "bert":
            encode = self.tokenizer.batch_encode_plus(
                contents,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True
            )
            cache_data = {
                "input_ids": encode["input_ids"],
                "attention_mask": encode["attention_mask"],
                "labels": torch.LongTensor(labels)
            }
        else:
            self.max_length = max(len(content) for content in contents)
            input_ids = torch.zeros((len(contents), self.max_length), dtype=torch.long)
            for i, content in enumerate(contents):
                vector = [self.vocab.get(word, self.vocab["[UNK]"]) for word in content][:self.max_length]
                input_ids[i, :len(vector)] = torch.LongTensor(vector)
            cache_data = {
                "input_ids": torch.LongTensor(input_ids),
                "labels": torch.LongTensor(labels)
            }
            torch.save(cache_data, self.cache_path)

    @property
    def vocab_size(self):
        return len(self.vocab) if self.vocab else self.tokenizer.vocab_size

    @property
    def num_classes(self):
        return len(self.label2id)
