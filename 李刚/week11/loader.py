import json

import torch
from torch.utils.data import DataLoader


class DataGenerator:
    def __init__(self, data_path, tokenizer, content_max_len, title_max_len):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.content_max_len = content_max_len
        self.title_max_len = title_max_len
        self.data = self.load()

    def load(self):
        data = []
        with open(self.data_path, encoding="utf8") as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                title = line["title"]
                content = line["content"]
                content_ids = self.tokenizer.encode(content, add_special_tokens=False, padding="max_length",
                                             max_length=self.content_max_len, truncation=True)
                title_ids = self.tokenizer.encode(title + '[SEP]', add_special_tokens=False, padding="max_length", max_length=self.title_max_len,
                                              truncation=True)
                input_ids = content_ids + [self.tokenizer.sep_token_id] + title_ids
                # print(input_ids)
                data.append(torch.LongTensor(input_ids))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_data(batch_size, data_path, tokenizer, content_max_len, title_max_len):
    dg = DataGenerator(data_path, tokenizer, content_max_len, title_max_len)
    dl = (DataLoader(dg, batch_size=batch_size, shuffle=True))
    return dl

if __name__ == '__main__':
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    dg = DataGenerator("sample_data.json", tokenizer, 120, 30)
    print(dg[0])