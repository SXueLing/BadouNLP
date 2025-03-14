import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from config import Config
import jieba
import json


# 加载数据的类
class Get_Data:
    def __init__(self, config, data_path, is_predict=False):
        self.config = config
        self.data_path = data_path
        self.is_predict = is_predict
        if self.config["model_type"] == "bert":
            self.config["max_length"] += 2
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
    def data_label(self):
        self.all_data = []
        with open(self.data_path, encoding="utf8") as f:
            sentences = f.read().split("\n\n")
            for sentence in sentences:
                data_x = []
                if self.config["model_type"] == "bert":
                    data_y = [-1]
                else:
                    data_y = []
                words = sentence.split("\n")
                for word in words:
                    if word.strip() == "":
                        continue
                    text, type = word.split()
                    data_x.append(text)
                    data_y.append(self.schema[str(type)])
                sentence_text = "".join(data_x)
                input_id = self.encode_sentence(sentence_text)
                output_id = self.padding(data_y, -1)
                if self.config["model_type"] == "bert":
                    output_id[-1] = -1
                self.all_data.append([torch.LongTensor(input_id), torch.LongTensor(output_id)])

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
        input_id = self.padding(input_id, self.vocab["padding"])
        return input_id

    def padding(self, input_id, padding_idx):
        input_id = input_id[:self.config["max_length"]]
        input_id += [padding_idx] * (self.config["max_length"] - len(input_id))
        return input_id

    # 加载数据
    def load(self):
        self.data_label()

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        return self.all_data[index]


# 提供预测和训练验证数据
def loader(data_path, config, shuffle=True):
    GD = Get_Data(config, data_path)
    DL = DataLoader(GD, batch_size=config["batch_size"], shuffle=shuffle)
    return DL


# 测试是否可用
if __name__ == "__main__":
    DL = loader(Config["eval_data_path"], Config)
    for index, batch_data in enumerate(DL):
        if torch.cuda.is_available():
            batch_data = [d.cuda() for d in batch_data]
        print(index)
        print(batch_data[0].shape)
        print(batch_data[1].shape)
        print(batch_data[0])
        print(batch_data[1])
        print("================================")
