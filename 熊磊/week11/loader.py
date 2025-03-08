import torch
from torch.utils.data import Dataset, DataLoader
from config import Config
import random
from transformers import BertTokenizer
import json

class DataGenerator(Dataset):
    def __init__(self, Config):
        super().__init__()
        self.config = Config
        self.data_path = Config['data_path']
        self.vocab_path = Config['vocab_path']
        self.tokenizer = BertTokenizer.from_pretrained(Config['bert_path'])
        self.vocab = load_vocab(Config)
        self.data = []
        self.load()
    
    def load(self):
        with open(self.data_path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                title, content = line["title"], line["content"]
                # title = self.tokenizer.encode(title, max_length=self.config['max_length'], pad_to_max_len=True)
                # content = self.tokenizer.encode(content, max_length=self.config['max_length'], pad_to_max_len=True)]
                
                title = [self.vocab.get(char, self.vocab['[UNK]']) for char in title]
                content = [self.vocab.get(char, self.vocab['[UNK]']) for char in content]
                x = title + [self.vocab.get('[SEP]')] + content 
                y = len(title) * [-1] + content + [-1]
                x = self.padding(x)
                y = self.padding(y)
                # print(len(title))
                # print(x)
                # print(y)
                # a = input()
                mask = torch.tril(torch.ones(self.config['max_length'], self.config['max_length']))
                mask[:len(title), :len(title)] = 1.0
                self.data.append([torch.LongTensor(x), torch.LongTensor(y), mask])

    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id
                        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(config):
    vocab = {}
    with open(config['vocab_path'], encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]
            vocab[char] = index
    return vocab


def build_dataset(Config):
    dg = DataGenerator(Config)
    dl = DataLoader(dg, batch_size=Config['batch_size'], shuffle=True)
    return dl

if __name__ == "__main__":
    # print(load_corpus(Config['data_path']))
    # vocab = load_vocab(Config['vocab_path'])
    # print(vocab['0'])
    # print(vocab['[CLS]'])
    # print(vocab['[PAD]'])
    # print(vocab['[UNK]'])

    # x, y = build_dataset(10, vocab, corpus)
    # print(generate_sample(vocab, corpus))
    # print(x.shape)
    mask = torch.ones(20, 20)
    mask[:10, :10] = 1.0
    print(mask)

    # mask = torch.tril(torch.ones(self.config['max_length'], self.config['max_length']))
    # mask[:len(title), :len(title)] = 1.0

    # dg = DataGenerator(Config)
    # dl = build_dataset(Config)
    # print(dg.data[0][0])
    # print(dg.data[0][1])
    pass
