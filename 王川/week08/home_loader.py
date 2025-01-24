import json
import jieba
import torch
import random
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import BertTokenizer


class Dataset:
    def __init__(self, data_path, config):
        self.path = data_path
        self.config = config
        self.schema = load_schema(config["schema_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        if config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.data_type = None
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.path, encoding = "utf8") as f:
            for line in f:
                line = json.loads(line)
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    target = line["target"]
                    for question in questions:
                        if self.config["model_type"] == "bert":
                            input = self.tokenizer.encode(question, max_length= self.config["max_length"], pad_to_max_length=True)
                        else:
                            input = self.encode_sentence(question)
                        input = torch.LongTensor(input)
                        self.knwb[self.schema[target]].append(input)
                else:
                    self.data_type = "test"
                    question, target = line
                    if self.config["model_type"] == "bert":
                        input = self.tokenizer.encode(question, max_length=self.config["max_length"],
                                                      pad_to_max_length=True)
                    else:
                        input = self.encode_sentence(question)
                    input = torch.LongTensor(input)
                    label = torch.LongTensor([self.schema[target]])
                    self.data.append([input, label])

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_sample_train()
        else:
            return self.data[index]

    def random_sample_train(self):
        standard_question_idx = list(self.knwb.keys())
        pos, neg = random.sample(standard_question_idx, 2)
        if len(self.knwb[pos]) < 2:
            return self.random_sample_train()
        else:
            a, p = random.sample(self.knwb[pos], 2)
            n = random.choice(self.knwb[neg])
        return [a, p, n]

    def encode_sentence(self, sentence):
        if self.config["vocab_path"] == 'word.txt':
            input = [self.vocab.get(word, self.vocab["[UNK]"]) for word in jieba.lcut(sentence)]
        else:
            input = [self.vocab.get(word, self.vocab["[UNK]"]) for word in sentence]
        input = self.padding(input)
        return input

    def padding(self, input):
        input = input[:self.config["max_length"]]
        input += [0] * (self.config["max_length"] - len(input))
        return input

def load_schema(schema_path):
    with open(schema_path, encoding = 'utf8') as f:
        return json.loads(f.read())

def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, encoding = 'utf8') as f:
        for index, line in enumerate(f):
            line = line[:-1]
            vocab[line] = index + 1
    return vocab

def load_data(data_path, config, shuffle = True):
    data = Dataset(data_path, config)
    data_loader = DataLoader(data, config["batch_size"], shuffle = shuffle)
    return data_loader
