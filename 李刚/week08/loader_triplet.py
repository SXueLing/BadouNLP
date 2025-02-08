
import json
import jieba
from collections import defaultdict
import torch
import random
from torch.utils.data import DataLoader


class LoadDatasetTriplet:
    def __init__(self, config, data_type):
        self.config = config
        self.train_data_path = config['train_data_path']
        self.valid_data_path = config['valid_data_path']
        self.batch_size = config['batch_size']
        self.word_vocab_path = config['word_vocab_path']
        self.char_vocab_path = config['char_vocab_path']
        self.vocab_type = config['vocab_type']
        self.vocab_dict = self.load_vocab_dict()
        self.max_length = config['max_length']
        self.label_path = config['label_path']
        self.label_dict = self.load_label_dict()
        self.data_type = data_type
        self.data_rate = config['data_rate']
        self.epoch_size = config['epoch_size']
        self.config["vocab_dict"] = len(self.vocab_dict)
        self.load(data_type)

    def __len__(self):
        if self.data_type == 'train':
            return self.epoch_size
        else:
            return len(self.eval_data)

    def __getitem__(self, item):
        if self.data_type == 'train':
            return self.train_sample_data()
        else:
            return self.eval_data[item]


    def train_sample_data(self):
        labels = self.knwb.keys()
        ap_l = random.choice(list(labels))
        if len(self.knwb[ap_l]) < 2:
            return self.train_sample_data()
        else:
            n_l = random.choice(list(labels))
            a, p = random.sample(self.knwb[ap_l], 2)
            n = random.choice(self.knwb[n_l])
        return [a, p, n]

    def load(self, data_type):
        self.knwb = defaultdict(list)
        self.eval_data = []
        if data_type == 'train':
            with open(self.train_data_path, encoding='utf8') as f:
                for line in f:
                    l = json.loads(line)
                    questions_list = l['questions']
                    target = l['target']
                    for question in questions_list:
                        question_index = self.load_question_to_index(question)
                        self.knwb[self.label_dict[target]].append(torch.LongTensor(question_index))
        elif data_type == 'evaluate':
            with open(self.valid_data_path, encoding='utf8') as f:
                for line in f:
                    eval_question = json.loads(line)
                    eval_question_index = self.load_question_to_index(eval_question[0])
                    self.eval_data.append([torch.LongTensor(eval_question_index), self.label_dict[eval_question[1]]])


    def load_vocab_dict(self):
        vocab_path = ''
        vocab_dict = {}
        if self.vocab_type == 'char':
            vocab_path = self.char_vocab_path
        elif self.vocab_type == 'word':
            vocab_path = self.word_vocab_path
        with open(vocab_path, encoding='utf8') as f:
            for i, v in enumerate(f):
                vocab_dict[v.replace('\n', '')] = i + 1
        return vocab_dict

    def load_label_dict(self):
        label_dict = {}
        with open(self.label_path, encoding='utf8') as f:
            for k,v in json.loads(f.read()).items():
                label_dict[k.replace('\n', '')] = v
        return label_dict

    def load_question_to_index(self, question):
        question_index = []
        vocab_dict = self.vocab_dict
        if self.vocab_type == 'char':
            for s in question:
                question_index.append(vocab_dict.get(s, vocab_dict.get('[UNK]')))
        elif self.vocab_type == 'word':
            words = jieba.lcut(question)
            for word in words:
                question_index.append(vocab_dict.get(word, vocab_dict.get('[UNK]')))
        question_index = self.padding_question(question_index)
        return question_index

    def padding_question(self, question_index):
        question_index = question_index[:self.max_length]
        question_index += [0] * (self.max_length - len(question_index))
        return question_index

def load_dataset(config, data_type):
    dase = LoadDatasetTriplet(config, data_type)
    dataset = DataLoader(dase,batch_size=config['batch_size'],shuffle=True)
    return dataset

if __name__ == '__main__':
    from config import Config
    da = load_dataset(Config, 'train')
    for i in da:
        print()
