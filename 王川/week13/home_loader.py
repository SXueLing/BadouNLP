import json
import torch
import jieba
from torch.utils.data import DataLoader
from transformers import BertTokenizer

class Dataset:
    def __init__(self, data_path, config):
        self.path = data_path
        self.config = config
        self.schema = load_schema(config["schema_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.sentences = []
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = [8]
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentence))
                input_ids = self.tokenizer.encode(sentence, truncation=True, max_length=self.config["max_length"], padding="max_length")
                labels = self.padding(labels, -1)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])

    def encode_sentence(self, sentence):
        if self.config["vocab_path"] == "word.txt":
            input_ids = [self.vocab.get(word, self.vocab["[UNK]"]) for word in jieba.cut(sentence)]
        else:
            input_ids = [self.vocab.get(word, self.vocab["[UNK]"]) for word in sentence]
        input_ids = self.padding(input_ids)
        return input_ids

    def padding(self, input_ids, pad=0):
        input_ids = input_ids[:self.config["max_length"]]
        input_ids += [pad] * (self.config["max_length"] - len(input_ids))
        return input_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.load(f)

def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]
            vocab[char] = index + 1
    return vocab

def load_data(data_path, config, shuffle=True):
    dataset = Dataset(data_path, config)
    data_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle)
    return data_loader
