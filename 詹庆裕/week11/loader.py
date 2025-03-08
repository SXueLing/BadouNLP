import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from config_file import Config
import json


class CustomDataset(Dataset):
    def __init__(self, tokenizer, dataset, max_length):
        self.tokenizer = tokenizer
        self.query = dataset["query"]
        self.answer = dataset["answer"]
        self.max_length = max_length

    def __len__(self):
        return len(self.query)

    def __getitem__(self, index):
        query = self.query[index]
        answer = self.answer[index]
        input_code = self.tokenizer.encode(query, add_special_tokens=True)
        label_encode = self.tokenizer.encode(answer, add_special_tokens=False)
        input_ids = input_code + label_encode + [self.tokenizer.sep_token_id]
        label = [-1] * (len(input_code)-1) + label_encode + [self.tokenizer.sep_token_id] + [-1]
        input_code = self.__padding_code(input_ids)
        target = self.__padding_code(label)
        mask = self.__build_mask(len(query), len(answer), self.max_length)
        return {
            "input_ids": torch.tensor(input_code, dtype=torch.long),
            "masks" : mask,
            "target": torch.tensor(target, dtype=torch.long)
        }

    def __padding_code(self, code):
        if len(code) > self.max_length:
            temp = code[:self.max_length]
        else:
            temp = code + [0] * (self.max_length - len(code))
        return temp

    def __build_mask(self, query_len, answer_len, max_len):
        enc_len = query_len + 2  # [CLS] + query + [SEP]
        dec_len = answer_len + 1  # answer + [SEP]
        total_len = enc_len + dec_len
        # 生成坐标矩阵
        i, j = torch.meshgrid(torch.arange(total_len), torch.arange(total_len), indexing='ij')
        # Encoder部分可见性（双向注意力）
        enc_mask = j < enc_len
        # Decoder部分可见性（因果注意力+关注Encoder）
        dec_mask = (i >= j) & (j >= enc_len)
        # 组合掩码
        mask = (enc_mask | dec_mask).int()
        # 处理Padding（超出实际长度的位置设为不可见）
        mask[total_len:] = 1
        mask[:, total_len:] = 0
        padding_mask = torch.zeros(max_len, max_len)
        index = min(total_len, max_len)
        padding_mask[0:index, 0:index] = mask[:index, :index]
        return padding_mask


def all_data():
    Knowledge_pair = {}
    query = []
    answer = []
    with open("news.json", "r", encoding="utf-8") as f:
         data = json.load(f)
    for line in data:
        query.append(line["title"])
        answer.append(line["content"])
    Knowledge_pair["query"] = query
    Knowledge_pair["answer"] = answer
    tokenizer = BertTokenizer.from_pretrained(Config["pretrain_model_path"])
    trainSet = CustomDataset(tokenizer, Knowledge_pair, Config["max_length"])
    train_pram = {
        "shuffle": True,
        "batch_size": Config["batch_size"]
    }
    train_loader = DataLoader(trainSet, **train_pram)
    return train_loader


if __name__ == "__main__":
    train_loader = all_data()
    for batch in train_loader:
        print(batch["input_ids"][0])
        print(batch["target"][0])
        print(batch["masks"][0])
        break


