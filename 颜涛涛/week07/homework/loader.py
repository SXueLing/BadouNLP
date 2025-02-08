import csv

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer


class DataGenerator:
    def __init__(self,data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        if self.config['model_type'] == 'bert':
            self.tokenizers = BertTokenizer.from_pretrained(self.config['pretrain_model_path'], return_dict=False)
        else:
            pass

        self.load()

    def load(self):
        self.data = []
        with open(self.path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for line in reader:

                inputs = line[1:]
                inputs_char = ''.join(inputs)
                if self.config['model_type'] == 'bert':
                    input_ids = self.tokenizers.encode(inputs_char, max_length=self.config['max_length'],
                                                       pad_to_max_length=True)
                else:
                    input_ids = self.encode_sentence(inputs_char)

                output_ids = int(line[0])
                input_codes = torch.LongTensor(input_ids)
                output_codes = torch.LongTensor([output_ids])
                self.data.append([input_codes, output_codes])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == '__main__':
    from config import Config

    # dg = DataLoader(Config)
    train_data = load_data(Config["train_data_csv_path"], Config)
    for index, data in enumerate(train_data):
        print(f"index: {index}, data: {data}")