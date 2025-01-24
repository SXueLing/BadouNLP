import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from config import Config
import csv


# 加载数据的类
class Get_Data:
    def __init__(self, config, data_path):
        self.config = config
        self.data_path = data_path
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(self.config["pretrain_model_path"])
        self.vocab = self.load_vocab(self.config["vocab_path"])
        self.load()

    # 做词表
    def load_vocab(self, vocab_path):
        vocab = {}
        vocab["padding"] = 0
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                vocab[line.strip()] = index + 1
            self.config["vocab_size"] = len(vocab)
        return vocab

    def data_to_dic(self):
        data_dict = {}
        with open(self.data_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                key = row[1]
                value = row[0]
                data_dict[key] = value
        return data_dict

    # 获得标签
    def get_data_y(self, data_dict):
        data_text = []
        self.data_y = []
        hash_lst = []
        for key, value in data_dict.items():
            self.data_y.append([float(value)])
            data_text.append(key)
            if value not in hash_lst:
                hash_lst.append(value)
        self.config["classify_num"] = len(hash_lst)
        return data_text

    # 准备输入embedding的索引数据
    def to_index(self, text):
        all_data = []
        for sentence in text:
            all_data.append(self.encode_sentence(sentence))
        return all_data

    def encode_sentence(self, sentence):
        if self.config["model_type"] == "bert":
            input_id = self.tokenizer.encode(sentence, max_length=self.config["max_length"], padding='max_length',
                                             truncation=True, return_attention_mask=True)
        else:
            input_id = []
            for char in sentence:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [self.vocab["padding"]] * (self.config["max_length"] - len(input_id))
        return input_id

    # 加载数据
    def load(self):
        data_dict = self.data_to_dic()
        data_text = self.get_data_y(data_dict)
        self.data_x = self.to_index(data_text)
        self.data = []
        for x, y in zip(self.data_x, self.data_y):
            self.data.append([torch.LongTensor(x), torch.LongTensor(y)])
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# 提供预测和训练验证数据
def loader(data_path, config, ispredict=False, shuffle=True):
    GD = Get_Data(config, data_path)
    if ispredict:
        DL = DataLoader(GD, batch_size=config["predict_num"], shuffle=shuffle)
    else:
        DL = DataLoader(GD, batch_size=config["batch_size"], shuffle=shuffle)
    return DL


# 测试是否可用
if __name__ == "__main__":
    dg = Get_Data(Config, "./dataset/eval.csv")
    DL = loader("./dataset/eval.csv", Config, ispredict=True)
    for index, batch_data in enumerate(DL):
        if torch.cuda.is_available():
            batch_data = [d.cuda() for d in batch_data]
        print(index)
        print(batch_data)
        x, _ = batch_data
        print(x)
        print("===============")
