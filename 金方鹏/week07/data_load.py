"""
数据加载
"""
import torch
import numpy as np
from transformers import BertTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class DataPreprocessing:
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        self.index_to_label = {0:"差评", 1:"好评"}
        self.label_to_index = dict((x,y) for x,y in self.index_to_label.items())
        # print(self.label_to_index)
        self.config["class_num"] = len(self.index_to_label) #类别数量
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"]) #处理词表
        self.config["vocab_size"] = len(self.vocab) #获取词表大小
        self.load()

    # 句子编码
    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        #print(input_id)
        return input_id

    # 按最大长度要求补齐截断
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))#不足最大长度补0
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


    def load(self):
        self.data = []
        dt = pd.read_csv(self.data_path) #pandas 读取数据
        text_length = dt["review"].apply(len) #文本长度总和
        avg_length = text_length.mean()  # 平均长度
        # print(avg_length)
        # 获取总样本数
        # num_samples = len(dt)
        # print(num_samples)
        # # 生成随机索引并打乱
        # indices = np.arange(num_samples)
        # np.random.shuffle(indices)  # 打乱索引
        # # 按比例划分
        # split = int(0.8 * num_samples)  # 80% 训练集，20% 测试集
        # train_indices = indices[:split]  # 训练集索引
        # test_indices = indices[split:]  # 测试集索引
        # # 划分数据集
        # train_data = dt.iloc[train_indices]  # 训练集
        # test_data = dt.iloc[test_indices]  # 测试集
        #print(len(train_data), len(test_data))
        # 保存训练集和测试集
        # train_data.to_csv('data/train_dataset.csv', index=False)
        # test_data.to_csv('data/test_dataset.csv', index=False)

        for index, row in dt.iterrows():  # 取出每一行数据（行索引，内容）
            # print(row["review"])
            label = row["label"]
            text = row["review"]
            if self.config["model_type"] == "bert":
                input_id = self.tokenizer.encode(text, max_length=self.config["max_length"], pad_to_max_length=True)
            else:
                input_id = self.encode_sentence(text)
            input_id = torch.LongTensor(input_id)
            label_index = torch.LongTensor([label])
            self.data.append([input_id, label_index])
        #print(self.data)
        return


#处理词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()  # 返回一个新字符串，移除了原字符串两端指定的字符，不指定则为空白字符。
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


def load_data(data_path, config, shuffle=True):
    dp = DataPreprocessing(data_path, config)
    dl = DataLoader(dp, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == '__main__':
    # print(load_vocab('chars.txt'))
    from config import Config
    dp = DataPreprocessing("data/test.csv", Config)
    # dp.load()
    # dp.encode_sentence("你好，你好看")
    # train_data = load_data('data/train_dataset.csv', Config)
    # for index, batch_data in enumerate(train_data):
    #     print(index, batch_data)