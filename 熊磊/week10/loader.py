import torch
from torch.utils.data import Dataset, DataLoader
from config import Config
import random
from transformers import BertTokenizer

class DataGenerator(Dataset):
    def __init__(self, Config):
        super().__init__()
        self.config = Config
        self.data_path = Config['data_path']
        self.vocab_path = Config['vocab_path']
        self.load()

def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

def load_vocab(path):
    vocab = {}
    with open(path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]
            vocab[char] = index
    return vocab

def generate_sample(vocab, corpus, window_size = 10):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    # x = [vocab['[CLS]']] + [vocab.get(char, vocab['[UNK]']) for char in corpus[start:end]]
    # y = [vocab.get(char, vocab['[UNK]']) for char in corpus[start + 1:end + 1]] + [vocab['[SEP]']]

    x = [vocab.get(char, vocab['[UNK]']) for char in corpus[start:end]]
    y = [vocab.get(char, vocab['[UNK]']) for char in corpus[start + 1:end + 1]]

    # x = corpus[start:end]
    # y = corpus[start + 1:end + 1]
    # tokenizer = BertTokenizer.from_pretrained(config['bert_path'])
    # x = tokenizer.encode(x, padding=False, add_special_tokens=False)
    # self.tokenizer.encode(sentenece, padding="max_length", max_length=self.config["max_length"], truncation=True, add_special_tokens=False)

    return x, y

def build_dataset(batch_size, vocab, corpus, window_size = None):
    data_x = []
    data_y = []
    for _ in range(batch_size):
        if window_size:
            x, y = generate_sample(vocab, corpus, window_size)
        else:
            x, y = generate_sample(vocab, corpus)
        data_x.append(x)
        data_y.append(y)
    return torch.LongTensor(data_x), torch.LongTensor(data_y)

if __name__ == "__main__":
    # print(load_corpus(Config['data_path']))
    vocab = load_vocab(Config['vocab_path'])
    # print(vocab['0'])
    # print(vocab['[CLS]'])
    # print(vocab['[PAD]'])
    # print(vocab['[UNK]'])
    corpus = load_corpus(Config['data_path'])
    x, y = build_dataset(10, vocab, corpus)
    print(generate_sample(vocab, corpus))
    print(x.shape)
