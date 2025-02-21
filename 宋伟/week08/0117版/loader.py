import json
import random
from torch.utils.data import Dataset, DataLoader

class TripletDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        self.triplets = self._create_triplets()

    def _create_triplets(self):
        triplets = []
        for entry in self.data:
            questions = entry['questions']
            target = entry['target']
            for anchor in questions:
                # 检查 questions 是否有其他选项作为 positive
                positive_candidates = [q for q in questions if q != anchor]
                if not positive_candidates:
                    continue  # 跳过无效的 Anchor
                
                positive = random.choice(positive_candidates)
                # Negative 示例
                negative_entry = random.choice([e for e in self.data if e['target'] != target])
                negative = random.choice(negative_entry['questions'])
                triplets.append((anchor, positive, negative))
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]

# 加载数据函数

def get_dataloader(data_path, batch_size):
    dataset = TripletDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)