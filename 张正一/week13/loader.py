import torch
from config import Config
import json
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset

class Loader:
    def __init__(self, config, data_path):
        self.data_path = data_path
        self.schema = self.load_schema(config['schema_path'])
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_model_path'])
        self.sentences = []
        self.data = []
        self.load()

    def load(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            segments = f.read().split('\n\n')
            # count = 0
            for segment in segments:
                # count += 1
                # if count == 10:
                #     break
                sentence = []
                labels = []
                for line in segment.split('\n'):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                sentence = ''.join(sentence)
                self.sentences.append(sentence)
                input_ids = self.encode_sentence(sentence)
                input_ids, labels = self.padding(input_ids, labels)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
                # print(torch.LongTensor(input_ids), torch.LongTensor(labels), torch.LongTensor(input_ids).shape, torch.LongTensor(labels).shape)

    def encode_sentence(self, sentence):
        input_ids = self.tokenizer.encode(sentence, max_length=Config['max_length'], pad_to_max_length=True)
        return input_ids[:-1]
    
    def padding(self, input_ids, labels):
        input_ids = input_ids[:Config['max_length']][1:]
        input_ids += [0] * (Config['max_length'] - len(input_ids))
        labels = labels[:Config['max_length']]
        labels += [Config['pad_label']] * (Config['max_length'] - len(labels))
        return input_ids, labels
    
    def load_schema(self, schema_path):
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
        
def load_data(config, data_path, shuffle=True):
    loader = Loader(config, data_path)
    data_loader = DataLoader(loader, batch_size=Config['batch_size'], shuffle=shuffle)
    return data_loader
    
load_data(Config, Config['data_path'])