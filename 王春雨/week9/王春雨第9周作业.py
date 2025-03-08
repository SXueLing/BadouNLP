###############  改用bert  实现命名实体识别 主要变化代码#############################

###############  loader.py 主要变化代码#############################
#loader.py 的 load 方法 增加了attention_mask   使用 BertTokenizer 作为encode
def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentenece = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentenece.append(char)
                    labels.append(self.schema[label])

                self.sentences.append("".join(sentenece))
                input_ids = self.tokenizer.encode(sentenece, max_length=self.config["max_length"], padding='max_length', truncation=True,add_special_tokens=True)
                labels    = self.padding(labels, -1)
                # 使用列表推导式生成 attention_mask
                attention_mask = [1 if id != 0 else 0 for id in input_ids]
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels), torch.LongTensor(attention_mask)])
        return

###############  model.py  主要变化代码##########################################
    #model.py   主要变化代码如下 使用 BertModel 采用交叉熵loss
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        self.bert  = BertModel.from_pretrained(config["bert_path"], return_dict=False)
        self.classify = nn.Linear(hidden_size , class_num)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失
      
    def forward(self, x, target=None, attention_mask=None):
        sequence_output, pooler_output = self.bert(x,attention_mask=attention_mask)
        x = self.classify(sequence_output)
        predict = x
        loss = None
        if target is not None:
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = predict.view(-1, predict.shape[-1])
                    active_labels = torch.where(
                        active_loss, target.view(-1), torch.tensor(self.loss.ignore_index).type_as(target)
                    )
                    loss = self.loss(active_logits, active_labels)
                return loss
        else:
                return predict
          
    ###############  main.py  主要变化代码##########################################
      # main.py  中训练时 传入 attention_mask
            input_id, labels,attention_mask = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id, labels,attention_mask=attention_mask)





########################################修改后按如下配置 测试效果如下#####################################################################
"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",
    "max_length": 128,
    "hidden_size": 768,
    "num_layers": 1,
    "epoch": 12,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 7e-4,
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"D:\八斗\课件\第六周 语言模型\bert-base-chinese"
}

2025-02-01 13:50:03,383 - __main__ - INFO - epoch 12 begin
2025-02-01 13:50:03,924 - __main__ - INFO - batch loss 0.017920
2025-02-01 13:50:28,978 - __main__ - INFO - batch loss 0.032817
2025-02-01 13:50:53,818 - __main__ - INFO - batch loss 0.116309
2025-02-01 13:50:53,818 - __main__ - INFO - epoch average loss: 0.035704
2025-02-01 13:50:53,818 - __main__ - INFO - 开始测试第12轮模型效果：
2025-02-01 13:50:58,171 - __main__ - INFO - PERSON类实体，准确率：0.743590, 召回率: 0.594872, F1: 0.660964
2025-02-01 13:50:58,171 - __main__ - INFO - LOCATION类实体，准确率：0.694561, 召回率: 0.680328, F1: 0.687366
2025-02-01 13:50:58,171 - __main__ - INFO - TIME类实体，准确率：0.843137, 召回率: 0.724719, F1: 0.779451
2025-02-01 13:50:58,171 - __main__ - INFO - ORGANIZATION类实体，准确率：0.484848, 召回率: 0.333333, F1: 0.395057
2025-02-01 13:50:58,171 - __main__ - INFO - Macro-F1: 0.630709
2025-02-01 13:50:58,171 - __main__ - INFO - Micro-F1 0.667666









##########################################完整代码如下####################################################
##########################################完整代码如下####################################################

#main.py
import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 获取迭代器
    #iterator = iter(train_data)

    # 从迭代器中获取第一个批次的数据
    #first_batch = next(iterator)
    # 打印第一个批次的数据
    #print(first_batch)
    #加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels,attention_mask = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id, labels,attention_mask=attention_mask)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return model, train_data

if __name__ == "__main__":
    model, train_data = main(Config)


#model.py
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        #self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        #self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.bert  = BertModel.from_pretrained(config["bert_path"], return_dict=False)
        self.classify = nn.Linear(hidden_size , class_num)
        #self.activation = torch.softmax     #sigmoid做激活函数
        #self.crf_layer = CRF(class_num, batch_first=True)
        #self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None, attention_mask=None):
        sequence_output, pooler_output = self.bert(x,attention_mask=attention_mask)
        x = self.classify(sequence_output)
        predict = x
        loss = None
        if target is not None:
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = predict.view(-1, predict.shape[-1])
                    active_labels = torch.where(
                        active_loss, target.view(-1), torch.tensor(self.loss.ignore_index).type_as(target)
                    )
                    loss = self.loss(active_logits, active_labels)
                return loss
        else:
                return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)


# loader.py

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentenece = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentenece.append(char)
                    labels.append(self.schema[label])

                self.sentences.append("".join(sentenece))
                input_ids = self.tokenizer.encode(sentenece, max_length=self.config["max_length"], padding='max_length', truncation=True,add_special_tokens=True)
                labels    = self.padding(labels, -1)
                # 使用列表推导式生成 attention_mask
                attention_mask = [1 if id != 0 else 0 for id in input_ids]
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels), torch.LongTensor(attention_mask)])
        return

    def encode_sentence(self, text, padding=True):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            input_id.append(self.tokenizer.encode(text, max_length=self.config["max_length"], pad_to_max_length=True))
                
            #for word in jieba.cut(text):
            #    input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)



#evaluate.py
# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data

"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)


    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        self.model.eval()
        for index, batch_data in enumerate(self.valid_data):
            sentences = self.valid_data.dataset.sentences[index * self.config["batch_size"]: (index+1) * self.config["batch_size"]]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels,attention_mask = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                pred_results = self.model(input_id,attention_mask=attention_mask) #不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results, sentences)
        self.show_stats()
        return

    def write_stats(self, labels, pred_results, sentences):
        assert len(labels) == len(pred_results) == len(sentences)
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            true_entities = self.decode(sentence, true_label)
            pred_entities = self.decode(sentence, pred_label)
            # print("=+++++++++")
            # print(true_entities)
            # print(pred_entities)
            # print('=+++++++++')
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

    def show_stats(self):
        F1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))
        self.logger.info("Macro-F1: %f" % np.mean(F1_scores))
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum([self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return

    '''
    {
      "B-LOCATION": 0,
      "B-ORGANIZATION": 1,
      "B-PERSON": 2,
      "B-TIME": 3,
      "I-LOCATION": 4,
      "I-ORGANIZATION": 5,
      "I-PERSON": 6,
      "I-TIME": 7,
      "O": 8
    }
    '''
    def decode(self, sentence, labels):
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results



                                 

