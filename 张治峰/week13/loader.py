from transformers import AutoTokenizer
import json
from torch.utils.data import DataLoader
import torch

def load_schema(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class DataGenerator:
    def __init__(self, data_path, config):
        self.path = data_path
        self.schema = load_schema(config["schema_path"])
        self.max_length = config["max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(config["pretrain_model_path"])
        self.data = []
        self.sentences = []
        self.load()

    def load(self):
        with open(self.path, encoding="utf-8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                labels = []
                sentence = ''
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence += char
                    labels.append(self.schema[label])
                input_ids = self.tokenizer.encode(sentence, max_length=self.max_length, padding="max_length",
                                                  truncation=True,add_special_tokens=False)
                self.sentences.append(sentence)
                labels = self.padding(labels)
                self.data.append([torch.LongTensor(input_ids),torch.LongTensor(labels)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def padding(self, input_ids, pad_token=-1):
        input_ids = input_ids[0:self.max_length]
        input_ids +=[pad_token] * (self.max_length - len(input_ids))
        return input_ids


def load_data(data_path,config,shuffle = True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg,shuffle=shuffle,batch_size=config["batch_size"])
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("ner_data/train", Config)
