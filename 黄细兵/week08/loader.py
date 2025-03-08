import json
import random
from collections import defaultdict

import jieba
import torch
from torch.utils.data import dataloader, DataLoader


class DataGenerator():
    def __init__(self, data_path, config):
        self.config = config
        self.data_path = data_path
        self.vocab = load_vocal(config['vocab_path'])
        self.schema = load_schema(config['schema_path'])
        self.config["vocab_size"] = len(self.vocab)
        self.train_data_size = config["epoch_data_size"]  # 由于采取随机采样，所以需要设定一个采样数量，否则可以一直采
        self.data_type = None
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.data_path, 'r', encoding='utf8') as f:
            for line in f:
                line = json.loads(line)
                #  训练集
                if isinstance(line, dict):
                    self.data_type = 'train'
                    questions = line['questions']
                    target = line['target']
                    for question in questions:
                        input_id = self.sentence_encodering(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[target]].append(input_id)
                # 验证集
                else:
                    self.data_type = 'test'
                    assert isinstance(line, list)
                    question, target = line
                    input_id = self.sentence_encodering(question)
                    input_id = torch.LongTensor(input_id)
                    label = torch.LongTensor([self.schema[target]])
                    self.data.append([input_id, label])
        return

    def __len__(self):
        if self.data_type == 'train':
            return self.train_data_size
        else:
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == 'train':
            return self.random_train_sample_v2()
        else :
            return self.data[index]

    # 随机生成样本数据
    def random_train_sample(self):
        questions_indexs = list(self.knwb.keys())
        rate = self.config['positive_sample_rate']
        if random.random() <= rate:
            p = random.choice(questions_indexs)
            if len(self.knwb[p]) < 2:
                return self.random_train_sample()
            else:
                u, v = random.sample(self.knwb[p], 2)
                return [u, v, torch.LongTensor([1])]
        else:
            n, n2 = random.sample(questions_indexs, 2)
            s1 = random.choice(self.knwb[n])
            s2 = random.choice(self.knwb[n2])
            return [s1, s2, torch.LongTensor([-1])]

    def random_train_sample_v2(self):
        standard_question_index = list(self.knwb.keys())
        p, n = random.sample(standard_question_index, 2)
        s1, s2 = random.sample(self.knwb[p], 2)
        # 随机一个负样本
        s3 = random.choice(self.knwb[n])
        # 前2个相似，后1个不相似，不需要额外在输入一个0或1的label，这与一般的loss计算不同
        return [s1, s2, s3]

    def sentence_encodering(self, question):
        input_id = []
        if self.config['vocab_path'] == 'words.text':
            for word in jieba.cut(question):
                input_id.append(self.vocab.get(word, self.vocab['[UNK]']))
        else:
            for char in question:
                input_id.append(self.vocab.get(char, self.vocab['[UNK]']))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.config['max_length']]
        input_id += [0] * (self.config['max_length'] - len(input_id))
        return input_id

def load_vocal(data_path):
    vocab_dict = {}
    with open(data_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            line = line.strip()
            vocab_dict[line] = index + 1
    return vocab_dict

def load_schema(schema_path):
    with open(schema_path, 'r', encoding='utf-8') as f:
          return json.load(f)

def load_data(data_path, config, shuffle=True):
    data = DataGenerator(data_path, config)
    dl = DataLoader(data, batch_size=config['batch_size'], shuffle=shuffle)
    return dl

if __name__ == '__main__':
    from config import Config
    # dg = load_data(Config["train_data_path"], Config)
    # print(dg[1])

    dg = DataGenerator(Config["valid_data_path"], Config)
    print(dg[1])
