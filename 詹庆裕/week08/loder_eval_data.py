import torch
from config import Config
from torch.utils.data import Dataset, DataLoader


class EvalDataset(Dataset):
    def __init__(self, path, vocab: dict):
        self.path = path
        self.seq_path = f"{path}"
        self.questions_to_label = {}
        self.vocab = vocab
        self.max_length = Config["max_length"]
        self._initialize_components()

    def _initialize_components(self):
        self.questions_to_label = {}  # 初始化字典
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                cleaned_line = line.strip().replace("[", "").replace("]", "")
                data = cleaned_line.split(",")
                question = data[0].strip().strip('"“”')  # 兼容中英文引号
                label = data[1].strip().strip('"“”')
                self.questions_to_label[question] = label

    def __len__(self):
        return len(self.questions_to_label.keys())

    def __getitem__(self, idx):
        question = list(self.questions_to_label.keys())[idx]
        vector = [self.vocab.get(word, self.vocab["[UNK]"]) for word in question][:self.max_length]
        vector.extend([0] * (self.max_length - len(vector)))
        label = self.questions_to_label[question]
        return {"question": torch.LongTensor(vector),
                "label": label}


class EvalDataLoad:
    def __init__(self, path, vocab: dict):
        self.dataset = EvalDataset(path, vocab)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size= Config["eval_batch_size"],
            shuffle=True,
            num_workers= Config["num_workers"],
        )

