import csv
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader

class Dataset:
    def __init__(self, data_path, config):
        self.path = data_path
        self.config = config
        if config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding = "utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for line in reader:
                label = int(line[0])
                input = line[1]
                if self.config["model_type"] == "bert":
                    #后面的补齐特别关键，别忘了
                    input = self.tokenizer.encode(input, max_length = self.config["max_length"], pad_to_max_length=True)
                else:
                    input = self.encode_sentence(input)
                input = torch.LongTensor(input)
                label = torch.LongTensor([label])
                self.data.append([input, label])

    def encode_sentence(self, sentence):
        input = []
        for char in sentence:
            input.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input = self.padding(input)
        return input

    def padding(self, input):
        input = input[:self.config["max_length"]]
        input += [0] * (self.config["max_length"] - len(input))
        return input

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, encoding = "utf-8") as f:
        for index, line in enumerate(f):
            char = line[:-1]
            vocab[char] = index + 1
    return vocab

def load_data(data_path, config, shuffle = True):
    data = Dataset(data_path, config)
    data_loader = DataLoader(data, batch_size = config["batch_size"], shuffle = shuffle)
    return data_loader
