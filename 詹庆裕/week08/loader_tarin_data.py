import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
from config import Config

class MyDataset(Dataset):
    def __init__(self, path, pos_neg_ratio:float = 0.5):
        self.pos_neg_ratio = pos_neg_ratio
        self.path = path
        self.tokenizer = {}
        self.max_length = Config["max_length"]
        self.label_groups = []
        self.all_questions = []
        self.label_map = {}
        self._initialize_components()

    def _initialize_components(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                self.label_groups.append({
                    "label": line["target"],
                    "questions": line["questions"]
                })
                for q in line["questions"]:
                    self.all_questions.append(q)
                    self.label_map[q] = line["target"]
        self.tokenizer = build_vocab()
        self._build_negative_index()

    def __len__(self):
        return len(self.all_questions)

    def _build_negative_index(self):
        self.negative_map = defaultdict(list)
        for q in self.all_questions:
            base_label = self.label_map[q]
            for group in self.label_groups:
                if group["label"] != base_label:
                    self.negative_map[q].extend(group["questions"])

    def _get_positive_example(self):
        group = random.choice(self.label_groups)
        text1, text2 = random.sample(group["questions"], 2)
        return (text1, text2), 1

    def _get_negative_example(self):
        base_q = random.choice(self.all_questions)
        neg_q = random.choice(self.negative_map[base_q])
        return (base_q, neg_q), -1

    def _get_triplet_example(self):
        (anchor, pos), _ = self._get_positive_example()
        neg = random.choice(self.negative_map[anchor])
        return anchor, pos, neg

    def encode(self, text):
        vector = [self.tokenizer.get(word, self.tokenizer["[UNK]"]) for word in text][:self.max_length]
        vector.extend([0] * (self.max_length - len(vector)))
        return vector


    def __getitem__(self, idx):
        if Config["loss_type"] == "cosEmbedding_loss":
            if random.random() < self.pos_neg_ratio:
                (text1, text2), label = self._get_positive_example()
            else:
                (text1, text2), label = self._get_negative_example()
            input_ids1 = self.encode(text1)
            input_ids2 = self.encode(text2)
            return {
                "input_ids1": torch.LongTensor(input_ids1),
                "input_ids2": torch.LongTensor(input_ids2),
                "labels": torch.tensor(label, dtype=torch.float)
            }
        else:
            anchor, pos, neg = self._get_triplet_example()
            anchor_encoder = self.encode(anchor)
            pos_encoder = self.encode(pos)
            neg_encoder = self.encode(neg)
            return {
                "anchor": torch.LongTensor(anchor_encoder),
                "pos": torch.LongTensor(pos_encoder),
                "neg": torch.LongTensor(neg_encoder)
            }



class MyDataLoad:
      def __init__(self, path):
          self.dataset = MyDataset(path)
          self.dataloader = DataLoader(
              self.dataset,
              batch_size=Config["batch_size"],
              shuffle=True,
              num_workers=Config["num_workers"],
              drop_last=True,
          )


def build_vocab():
    with open(Config["vocab_path"], "r", encoding="utf-8") as f:
        vocab = {word.strip(): idx + 1 for idx, word in enumerate(f)}
        return vocab
