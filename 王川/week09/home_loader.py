import json
import torch
import jieba
from torch.utils.data import DataLoader
from transformers import BertTokenizer


class Dataset:
    def __init__(self, data_path, config):
        self.path = data_path
        self.config = config
        self.vocab = load_vocab(config["vocab_path"])
        self.schema = load_schema(config["schema_path"])
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
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":      #提高代码健壮性,防止句子中间出现意外"\n"
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentence))
                # input = self.encode_sentence(sentence)
                input = self.tokenizer.encode("".join(sentence), max_length= self.config["max_length"], pad_to_max_length=True)
                labels = [-1] + labels + [-1]
                labels = self.padding(labels, -1)
                self.data.append([torch.LongTensor(input), torch.LongTensor(labels)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def encode_sentence(self, sentence):
        if self.config["vocab_path"] == "word.txt":
            input = [self.vocab.get(word, self.vocab["[UNK]"]) for word in jieba.cut(sentence)]
        else:
            input = [self.vocab.get(word, self.vocab["[UNK]"]) for word in sentence]
        input = self.padding(input)
        return input

    def padding(self, input, padding = 0):
        input = input[:self.config["max_length"]]
        input += [padding] * (self.config["max_length"] - len(input))
        return input

def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            line = line.strip()
            vocab[line] = index + 1
    return vocab


def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.load(f)

def load_data(data_path, config, shuffle = True):
    dataset = Dataset(data_path, config)
    data_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle)
    return data_loader

