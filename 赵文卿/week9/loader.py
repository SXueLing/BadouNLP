'''
Author: Zhao
Date: 2025-01-20 20:34:45
LastEditTime: 2025-01-22 13:08:01
FilePath: loader.py
Description: 数据加载

'''
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import json
import jieba
from transformers import BertTokenizer
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.config["vocab_size"] = self.tokenizer.vocab_size  # 更新配置中的词汇表大小
        self.sentences = []  # 保存所有句子
        self.schema = self.load_schema(config["schema_path"])  # 加载标签模式
        self.load()  # 加载数据

    def load(self):
        self.data = []
        # 打开并读取数据文件
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")  # 按双换行分段
            for segment in segments:
                labels = []
                sentence = []
                for line in segment.split("\n"):  # 按单换行分行
                    if line.strip() == "":
                        continue  # 跳过空行
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])  # 转换标签为数值
                    # 添加 [CLS] 和 [SEP] 标记
                    sentence = ["[CLS]"] + sentence + ["[SEP]"]
                    labels = [-1] + labels + [-1]
                self.sentences.append("".join(sentence))
                input_ids = self.tokenizer.encode(sentence, max_length=self.config["max_length"], padding='max_length', truncation=True)
                labels = self.padding(labels, pad_token=-1)  # 对标签进行填充
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])  # 保存编码后的输入和标签
        # logger.info("Sample Labels: %s", labels[:10])
        # logger.info("Sample Labels end: %s", labels[-10:])
        return

    def encode_sentence(self, sentence, padding=True):
        input_id = []
        for char in sentence:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"])) # 获取字符的索引，找不到时用 [UNK] 代替
            
        input_id = self.padding(input_id) # 对序列进行填充
        return input_id

    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]  # 截断输入
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))  # 填充到最大长度
        return input_id

    def __len__(self):
        return len(self.data)  # 返回数据集大小

    def __getitem__(self, index):
        return self.data[index]  # 根据索引返回数据

    def load_schema(self, path):
        # 加载标签模式
        with open(path, encoding="utf8") as f:
            return json.load(f)

def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)  # 创建数据生成器实例
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)  # 创建数据加载器
    return dl

# if __name__ == "__main__":
#     from config import Config
#     dg = DataGenerator("week9/data/train", Config)
    
#     print(dg[0])