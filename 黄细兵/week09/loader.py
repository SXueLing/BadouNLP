import json
import jieba
import torch
from torch.utils.data import dataloader, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""

class DataGenerator:
    def __init__(self, data_path, config):
        self.path = data_path
        self.config = config
        # self.vocab = load_vocab(config['vocab_path'])
        self.tokenizer = load_vocab_bert(config["bert_path"])
        self.schema = load_schema(config['schema_path'])
        # self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.load()

    def load(self):
        self.data = []
        with open(self.path, 'r', encoding='utf-8') as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentences = []
                labels = [8]
                for sentence in segment.split("\n"):
                    if sentence == '':
                        continue
                    sen, label = sentence.split()
                    sentences.append(sen)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentences))
                # 句子转vocab
                input_id = self.encode_sentence_bert(sentences)
                # label转schema
                labels = self.padding(labels, -1)
                self.data.append([torch.LongTensor(input_id), torch.LongTensor(labels)])
        return

    def encode_sentence_bert(self, text, padding=True):
        return self.tokenizer.encode(text, padding="max_length", max_length=self.config["max_length"], truncation=True)

    def encode_sentence(self, sentences, padding=True):
        input_id = []
        vocab_path = self.config['vocab_path']
        if vocab_path == 'chars.txt':
            for char in sentences:
                input_id.append(self.vocab.get(char, self.vocab['[UNK]']))
        else:
            for word in jieba.cut(sentences):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id

    def padding(self, input_ids, pad_token=0):
        input_ids = input_ids[:self.config['max_length']]
        input_ids += [pad_token] * (self.config['max_length'] - len(input_ids))
        return input_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_vocab( vocab_path):
    vocab_dict = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            line = line.strip()
            vocab_dict[line] = index + 1
    return vocab_dict

def load_vocab_bert(vocab_path):
    return BertTokenizer.from_pretrained(vocab_path)

def load_schema( schema_path):
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_data(path, config, shuffle=True):
    data = DataGenerator(path, config)
    dl = DataLoader(data, batch_size=config['batch_size'], shuffle=True)
    return dl


if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("ner_data/train", Config)
    print(dg.data)
