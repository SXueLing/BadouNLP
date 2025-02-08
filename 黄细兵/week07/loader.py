from transformers import BertTokenizer
import torch
import csv
import json
from torch.utils.data import Dataset, DataLoader
from config import Config


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '差评', 1: '好评'}
        self.config['class_num'] = len(self.index_to_label)
        if self.config['model_type'] == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(config['pretrain_model_path'])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()

    def load(self):
        self.date_list = []   # 8390 * 30
        with open(self.path, encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                label = int(line['label'])
                text = line['review']
                if self.config['model_type'] == 'bert':
                    input_id = self.tokenizer.encode(text, max_length=self.config["max_length"], pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(text)
                input_id = torch.LongTensor(input_id)
                input_label = torch.LongTensor([label])
                self.date_list.append((input_id, input_label))
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab['[UNK]']))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.date_list)

    def __getitem__(self, index):
        return self.date_list[index]



def load_vocab(vocab_path):
    dict = {}
    with open(vocab_path, encoding='utf-8') as f:
        for index, text in enumerate(f):
            token = text.strip()
            dict[token] = index + 1
    return dict


def csv_to_json(csv_path, train_json_path, valid_json_path, rate):
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    train_total = int(len(data) * rate)

    train_data = []
    valid_data = []
    for d in data:
        if len(train_data) < train_total:
            train_data.append(d)
        else:
            valid_data.append(d)

    print("json总数 %s" % len(data))  # 11987
    print("train_json总数 %s" % len(train_data))  # 8390
    print("valid_json总数 %s" % len(valid_data))  # 3597

    with open(train_json_path, 'w', encoding='utf-8') as json_file:
        # json.dump(train_data, json_file, indent=4, ensure_ascii=False)
        for train_datum in train_data:
            json_str = json.dumps(train_datum, ensure_ascii=False)
            json_file.write(json_str + '\n')

    with open(valid_json_path, 'w', encoding='utf-8') as json_file:
        # json.dump(valid_data, json_file, indent=4, ensure_ascii=False)
        for valid_datum in valid_data:
            json_str = json.dumps(valid_datum, ensure_ascii=False)
            json_file.write(json_str + '\n')

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl  # batch_size * voc_len * max_len  128*8390*30

if __name__ == '__main__':
    # csv_to_json('./csv/example.csv', './data/train.json', './data/valid.json', 0.7)
    dl = load_data("./data/train.json", Config, shuffle=True)
    print(dl[0])
