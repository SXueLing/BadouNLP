'''
pass
'''
import csv
from collections import Counter

import torch.utils.data
import config
import numpy as np



class Dataset():
    def __init__(self, train_data_size=1):
        vocab_path = config.config['vocab_path']
        data_path = config.config['data_path']
        self.vocab = self.get_vocab(vocab_path)
        self.data = self.open_data(data_path, self.vocab, train_data_size)
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def open_data(self, path, vocab, train_data_size, padding_length=50):
        data = []
        with open(path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)

            # 逐行数据处理加入 data 中
            for data_type, data_content in reader:
                data_content = [vocab.get(word, vocab['[UNK]']) for word in data_content]
                if len(data_content) > padding_length:
                    # 超出长度截断
                    data_content = data_content[: padding_length]
                else:
                    # 不足长度补齐
                    while len(data_content) < padding_length:
                        data_content.append(vocab['pad'])
                # 调整输出内容类型
                data_content = torch.LongTensor(data_content)
                if data_type == '1':
                    data_type = torch.FloatTensor([1])
                else:
                    data_type = torch.FloatTensor([0])
                data.append([data_content, data_type])
                self.all_data = data

        # 按照百分比取出数据作为训练数据
        if train_data_size != 1:
            self.train_data = []
            self.test_data = []
            for each in data:
                if np.random.random(1) < train_data_size:
                    self.train_data.append(each)
                else:
                    self.test_data.append(each)

        return data

    def get_vocab(self, path):
        vocab = {"pad": 0}
        with open(path, mode='r', encoding='utf-8') as file:
            index = 1
            for line in file:
                new_line = line.strip()
                vocab[new_line] = index
                index += 1
        vocab['unk'] = len(vocab)
        return  vocab

    def change_data(self, mode='train'):
        if mode == 'train':
            self.data = self.train_data
        elif mode == 'test':
            self.data = self.test_data
        else:
            self.data = self.all_data
        return self.data



if __name__ == '__main__':
    # vocab_path =
    # data_path =
    data = Dataset(0.8)
    train_data = data.change_data()
    test_data = data.change_data('test')
    all_data = data.change_data('all')

