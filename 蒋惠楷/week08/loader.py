import json
from collections import defaultdict
import torch.utils
from torch.utils.data import DataLoader
from config import Config
import jieba
import torch
import random

'''加载数据'''

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] =len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        self.train_data_size = config["epoch_data_size"] # 由于采取随机采样，所以需要设定一个采样数量，否则会一直采
        self.data_type = None # 用来标识加载的数据是 "train" or "test"
        self.load()

    '''加载训练集或测试集'''
    def load(self):
        self.data = []
        self.knwb = defaultdict(list) # 存储训练集中每个标签对应的文本序列
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                # 训练集的数据是以字典存储的
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    target = line["target"]
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[target]].append(input_id)
                # 测试集的数据是以列表存储的
                else:
                    self.data_type = 'test'
                    assert isinstance(line, list)
                    question, target = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    target_index = torch.LongTensor([self.schema[target]])
                    self.data.append([input_id, target_index])
        return
    
    '''文本编码'''
    def encode_sentence(self, text):
        input_id = []
        # 以词为单位对文本进行分词
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.lcut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            # 以字为单位对文本进行分词
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
                # print(f"字符: {char}, 编码: {input_id}")
        input_id = self.padding(input_id)
        return input_id
    
    '''补齐或截断输入的序列, 使其可以在一个batch内运算'''
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id
    
    '''获取数据集大小'''
    def __len__(self):
        # 如果是训练集，返回配置文件中 epoch_data_size 指定的训练样本数量
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
        # 如果是测试集，返回测试集中的样本数量
            assert self.data_type == "test", self.data_type
            return len(self.data)

    '''获取数据项'''
    def __getitem__(self, index):
        # 如果是训练集，调用 random_train_sample()
        if self.data_type == "train":
            return self.random_train_triplet_loss() # 随机生成一个训练样本
        else:
            # 如果是测试集，返回索引对应的测试样本
            return self.data[index]
        
    '''Triplet Loss 训练样本生成'''
    def random_train_triplet_loss(self):
        standard_question_index = list(self.knwb.keys())
        # 随机选择一个问题
        p = random.choice(standard_question_index)
        
        # 如果标准问下不足两个问题，则无法选取，所以重新随机一次
        if len(self.knwb[p]) < 2:
            return self.random_train_triplet_loss()
        else:
            # 随机选择正样本和负样本
            anchor = random.choice(self.knwb[p])  # 锚点
            positive = random.choice(self.knwb[p])  # 正样本
            # 选择不同问题的负样本
            negative_p = random.choice([q for q in standard_question_index if q != p])
            negative = random.choice(self.knwb[negative_p])  # 负样本
            
            # 返回三元组
            return [anchor, positive, negative]



'''加载字表或词表'''
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf-8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1 # 0留给padding
    return token_dict

'''加载数据匹配标签schema'''
def load_schema(schema_path):
    with open(schema_path, encoding="utf-8") as f:
        return json.loads(f.read())

'''使用torch自带的DataLoader类封装数据'''
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    knwb_data = load_data(Config["valid_data_path"], Config)
