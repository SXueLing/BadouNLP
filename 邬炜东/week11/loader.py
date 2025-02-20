import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from config import Config
import json
import numpy as np


# 加载数据的类
class Get_Data:
    def __init__(self, config, data_path):
        self.config = config
        self.data_path = data_path
        self.config["max_length_content"] += 2
        self.config["max_length_title"] += 2
        self.tokenizer = BertTokenizer.from_pretrained(self.config["pretrain_model_path"])
        self.vocab = self.load_vocab(self.config["vocab_path"])
        self.load()

    # 做词表
    def load_vocab(self, vocab_path):
        vocab = {}
        vocab["padding"] = 0
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                vocab[line.strip()] = index + 1
            self.config["vocab_size"] = len(vocab)
        return vocab

    # 文本转索引
    def encode_sentence(self, sentence, object):
        input_id = self.tokenizer.encode(sentence, max_length=object, padding='max_length',
                                         truncation=True, return_attention_mask=True)
        return input_id

    def padding(self, input_id, padding_idx):
        input_id = input_id[:self.config["max_length"]]
        input_id += [padding_idx] * (self.config["max_length"] - len(input_id))
        return input_id

    def load_corpus(self, corpus_path):
        corpus = ""
        with open(self.config["train_data_path"], encoding="gbk") as f:
            for line in f:
                corpus += line.strip()
        return corpus

    # 加载数据
    def load(self):
        with open(self.config["train_data_path"], encoding="utf8") as f:
            news_list = json.load(f)
        self.data_ids = []
        for news in news_list:
            title = self.encode_sentence(news["title"], self.config["max_length_title"])
            content = self.encode_sentence(news["content"], self.config["max_length_content"])
            bridge_token = self.tokenizer.encode(self.config["bridge_token"],
                                                 add_special_tokens=False)
            x_id = content[:-1] + bridge_token + title[1:]
            y_id = [0] * (self.config["max_length_content"] + len(self.config["bridge_token"]) - 1) + title[1:]
            y_id = y_id[1:] + [0]
            self.data_ids.append([torch.LongTensor(x_id), torch.LongTensor(y_id)])

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, index):
        return self.data_ids[index]


# 提供预测和训练验证数据
def loader(data_path, config, shuffle=False):
    GD = Get_Data(config, data_path)
    DL = DataLoader(GD, batch_size=config["batch_size"], shuffle=shuffle)
    return DL

def generate_mask(batch_size):
    total = Config["max_length_content"] + len(Config["bridge_token"]) + Config["max_length_title"] - 2
    mask = np.zeros((total, total))
    for i in range(Config["max_length_content"] + len(Config["bridge_token"]) - 1):
        for j in range(Config["max_length_content"] + len(Config["bridge_token"]) - 1):
            mask[i][j] = 1
    for i in range(Config["max_length_content"] + len(Config["bridge_token"]) - 1, total):
        for j in range(i + 1):
            mask[i][j] = 1
    mask = torch.LongTensor(mask)
    mask = mask.repeat(batch_size, 1, 1)
    return mask.cuda()

# 测试是否可用
if __name__ == "__main__":
    DL = loader(Config["train_data_path"], Config)
    mask = generate_mask(1)
    print(mask)
    print(mask.shape)
    for index, batch_data in enumerate(DL):
        if torch.cuda.is_available():
            batch_data = [d.cuda() for d in batch_data]
        print(index)
        print(batch_data[0][-1])
        print(batch_data[1][-1])
        print(batch_data[0].shape)
        print(batch_data[1].shape)
        print("================================")
