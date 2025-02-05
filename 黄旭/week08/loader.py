import json
from collections import defaultdict
import os
import jieba
import torch
import random
import numpy as np
from torch.utils.data import Dataset ,  DataLoader


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.data_path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        self.train_data_size = config["epoch_data_size"]
        self.data_type = None
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.data_path,encoding = "utf8")as f:
            for index ,line in enumerate(f):
            #for line in f:
                line = json.loads(line)
                if isinstance(line,dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                else:
                    self.data_type = "test"
                    assert isinstance(line,list)
                    question , label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id,label_index])
        return


    def encode_sentence(self,text):
        # 定义一个空列表，用于存储输入的id
        input_id = []
        # 判断配置文件中的词汇表路径
        if self.config["vocab_path"] == "words.txt":
            # 如果词汇表路径为words.txt，则使用jieba分词，将文本中的每个词转换为id
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word,self.vocab["[UNK]"]))
        else:
            # 如果词汇表路径不是words.txt，则将文本中的每个字符转换为id
            for char in text:
                input_id.append(self.vocab.get(char,self.vocab["[UNK]"]))
        # 对输入的id进行填充
        input_id = self.padding(input_id)
        # 返回填充后的id
        return input_id

    def padding(self,input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id
    
    def __len__(self):
        if self.data_type =="train":
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test",self.data_type
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type =="train":
            return self.random_train_sample()
        else:
            return self.data[index]

    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())
        p1,p2 = random.sample(standard_question_index,2)
        if len(self.knwb[p1])<2:
            s1 = s2 = random.choice(self.knwb[p1])
        else:
            s1,s2 = random.sample(self.knwb[p1],2)
        s3 = random.choice(self.knwb[p2])
        return [s1,s2,s3]

    # def random_train_sample(self):
    #     # 获取标准问题的索引列表
    #     standard_question_index = list(self.knwb.keys())
    #     # 如果随机数小于等于正样本率，则选择一个标准问题
    #     if random.random() <= self.config["positive_sample_rate"]:
    #         p = random.choice(standard_question_index)
    #         # 如果该标准问题对应的答案少于2个，则重新选择一个标准问题
    #         if len(self.knwb[p])<2:
    #             return self.random_train_sample()
    #         # 否则，从该标准问题的答案中随机选择2个，并返回
    #         else:
    #             s1,s2 = random.sample(self.knwb[p],2)
    #             return [s1,s2,torch.LongTensor([1])]
    #     # 否则，随机选择2个标准问题
    #     else:
    #         p,n = random.sample(standard_question_index,2)
    #         # 从第一个标准问题的答案中随机选择一个
    #         s1 = random.choice(self.knwb[p])
    #         # 从第二个标准问题的答案中随机选择一个
    #         s2 = random.choice(self.knwb[n])
    #         return [s1,s2,torch.LongTensor([-1])]
            
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path,encoding="utf8") as f:
        for index,line in enumerate(f):
             token = line.strip()
             token_dict[token] = index + 1 #0留给padding
    return token_dict

def load_schema(schema_path):
    with open(schema_path,encoding="utf-8") as f:
        return json.loads(f.read())

def load_data(data_path,config,shuffle=True):
    dg = DataGenerator(data_path,config)
    dl = DataLoader(dg,batch_size=config["batch_size"],shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    #Config["max_length"] = 20
    #Config["batch_size"] = 2
    #dg = DataGenerator("./data/valid.json",Config)
    #print (dg[-1])
    dg = load_data("./data/train.json",Config)
    #dg = DataGenerator("./data/valid.json",Config)
    #print (dg[0])
    for index,batch_data in enumerate(dg):
    #    print(batch_data)
        print (type(batch_data))
        print (len(batch_data))
    #for i in dg:
    #    print(i)

