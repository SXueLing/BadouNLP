import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from config import Config
import jieba
import json
import random
from _collections import defaultdict


# 加载数据的类
class Get_Data:
    def __init__(self, config, data_path, is_predict=True):
        self.config = config
        self.data_path = data_path
        self.is_predict = is_predict
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(self.config["pretrain_model_path"])
        self.vocab = self.load_vocab(self.config["vocab_path"])
        self.schema = self.schema_load(config["schema_data_path"])
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

    # schema是schema文件的字典格式
    def schema_load(self, schema_path):
        with open(schema_path, encoding='utf-8') as f:
            schema = json.loads(f.read())
            return schema

    # data_list列表里的元素是train里的每一行，data_dict是train中{target:所有问题的索引}
    def data_list(self):
        data_list = []
        self.data_dict = defaultdict(list)
        with open(self.data_path, encoding="utf8") as f:
            for line in f:
                data_line = json.loads(line)
                if self.is_predict:
                    data_line[0] = self.encode_sentence(data_line[0])
                    data_line[1] = self.schema[data_line[1]]
                    data_list.append([torch.LongTensor(data_line[0]), torch.LongTensor([data_line[1]])])
                else:
                    for index, sentence in enumerate(data_line["questions"]):
                        encode_sentence = self.encode_sentence(sentence)
                        data_line["questions"][index] = encode_sentence
                    data_line["target"] = self.schema[data_line["target"]]
                    data_list.append(data_line)
                    self.data_dict[data_line["target"]] = data_line["questions"]
        return data_list

    # 文本转索引
    def encode_sentence(self, sentence):
        if self.config["model_type"] == "bert":
            input_id = self.tokenizer.encode(sentence, max_length=self.config["max_length"], padding='max_length',
                                             truncation=True, return_attention_mask=True)
        else:
            input_id = []
            if self.config["vocab_path"] == "words.txt":
                sentence = jieba.lcut(sentence)
            for char in sentence:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [self.vocab["padding"]] * (self.config["max_length"] - len(input_id))
        return input_id

    # 随机从data_list里抽取正负样本
    def pos_neg_get(self, data_list):
        pos_index, neg_index = random.sample(range(len(data_list)), 2)
        if len(data_list[pos_index]["questions"]) < 2:
            return self.pos_neg_get(data_list)
        else:
            pos_1, pos_2 = random.sample(data_list[pos_index]["questions"], 2)
        neg = random.sample(data_list[neg_index]["questions"], 1)
        return pos_1, pos_2, neg

    # 加载数据
    def load(self):
        data_list = self.data_list()
        if self.is_predict:
            return data_list
        else:
            pos_1, pos_2, neg = self.pos_neg_get(data_list)
            return [torch.LongTensor(pos_1), torch.LongTensor(pos_2), torch.LongTensor(neg)]

    def __len__(self):
        if self.is_predict:
            return self.config["predict_num"]
        else:
            return self.config["train_num"]

    def __getitem__(self, index):
        if self.is_predict:
            return self.data_list()[index]
        else:
            return self.load()


# 提供预测和训练验证数据
def loader(data_path, config, is_predict=False, shuffle=True):
    GD = Get_Data(config, data_path, is_predict=is_predict)
    DL = DataLoader(GD, batch_size=config["batch_size"], shuffle=shuffle)
    if is_predict:
        return DL
    else:
        return DL, GD.data_dict


# 测试是否可用
if __name__ == "__main__":
    # DL= loader("./data/valid.json", Config, is_predict=True)
    DL, data_dict = loader("./data/train.json", Config)
    for index, batch_data in enumerate(DL):
        if torch.cuda.is_available():
            batch_data = [d.cuda() for d in batch_data]
        print(index)
        print(batch_data[0].shape)
        print(batch_data[1].shape)
        print(len(batch_data))
        print("================================")
