Config = {
    "model_path": "model_output",
    "schema_path": "../data/schema.json",
    "train_data_path": "../data/train.json",
    "valid_data_path": "../data/valid.json",
    "vocab_path":"../chars.txt",
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 10,
    "batch_size": 32,
    "epoch_data_size": 200,     #每轮训练中采样数量
    "positive_sample_rate":0.5,  #正样本比例
    "optimizer": "adam",
    "learning_rate": 1e-3,
}

"""
数据加载
"""
import json
from collections import defaultdict
import random

import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
from transformers import BertTokenizer


# 获取字表集
def load_vocab(path):
    vocab = {}
    with open(path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            word = line.strip()
            # 0留给padding位置，所以从1开始
            vocab[word] = index + 1
        vocab['unk'] = len(vocab) + 1
    return vocab


# 数据预处理 裁剪or填充
def padding(input_ids, length):
    if len(input_ids) >= length:
        return input_ids[:length]
    else:
        padded_input_ids = input_ids + [0] * (length - len(input_ids))
        return padded_input_ids


# 文本预处理
# 转化为向量
def sentence_to_index(text, length, vocab):
    input_ids = []
    for char in text:
        input_ids.append(vocab.get(char, vocab['unk']))
    # 填充or裁剪
    input_ids = padding(input_ids, length)
    return input_ids


class DataGenerator:
    def __init__(self, data_path, config):
        # 加载json数据
        self.load_know_base(config["train_data_path"])
        self.schema = self.load_schema(config["schema_path"])
        self.data_path = data_path
        self.config = config
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["bert_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.train_flag = None
        self.load_data()

    def __len__(self):
        if self.train_flag:
            return self.config["simple_size"]
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.train_flag:
            # return self.random_train_sample()  # 随机生成一个训练样本
            # triplet loss:
            return self.random_train_sample_for_triplet_loss()
        else:
            return self.data[idx]

    def load_data(self):
        self.train_flag = self.config["train_flag"]
        dataset_x = []
        dataset_y = []
        self.knwb = defaultdict(list)
        if self.train_flag:
            for target, questions in self.target_to_questions.items():
                for question in questions:
                    input_id = sentence_to_index(question, self.config["max_len"], self.vocab)
                    input_id = torch.LongTensor(input_id)
                    # self.schema[target]
                    self.knwb[self.schema[target]].append(input_id)
        else:
            with open(self.data_path, encoding="utf8") as f:
                for line in f:
                    line = json.loads(line)
                    assert isinstance(line, list)
                    question, target = line
                    input_id = sentence_to_index(question, self.config["max_len"], self.vocab)
                    # input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[target]])
                    # self.data.append([input_id, label_index])
                    dataset_x.append(input_id)
                    dataset_y.append(label_index)
                self.data = Data.TensorDataset(torch.tensor(dataset_x), torch.tensor(dataset_y))
        return

    def load_know_base(self, know_base_path):
        self.target_to_questions = {}
        with open(know_base_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                content = json.loads(line)
                questions = content["questions"]
                target = content["target"]
                self.target_to_questions[target] = questions
        return

    def load_schema(self, param):
        with open(param, encoding="utf8") as f:
            return json.loads(f.read())

    def random_train_sample(self):
        target = random.choice(list(self.knwb.keys()))
        # 随机正样本：
        # 随机正样本
        if random.random() <= self.config["positive_sample_rate"]:
            if len(self.knwb[target]) <= 1:
                return self.random_train_sample()
            else:
                question1 = random.choice(self.knwb[target])
                question2 = random.choice(self.knwb[target])
                # 一组
                # dataset_x.append([question1, question2])
                # # 二分类任务 同一组的question target = 1
                # dataset_y.append([1])
                return [question1, question2, torch.LongTensor([1])]
        else:
            # 随机负样本：
            p, n = random.sample(list(self.knwb.keys()), 2)
            question1 = random.choice(self.knwb[p])
            question2 = random.choice(self.knwb[n])
            # dataset_x.append([question1, question2])
            # dataset_y.append([-1])
            return [question1, question2, torch.LongTensor([-1])]

    # triplet_loss随机生成3个样本 锚样本A, 正样本P, 负样本N
    def random_train_sample_for_triplet_loss(self):
        target = random.choice(list(self.knwb.keys()))
        # question1锚样本 question2为同一个target下的正样本 question3 为其他target下样本
        question1 = random.choice(self.knwb[target])
        question2 = random.choice(self.knwb[target])
        question3 = random.choice(self.knwb[random.choice(list(self.knwb.keys()))])
        return [question1, question2, question3]

def load_data_batch(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    if config["train_flag"]:
        dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    else:
        dl = DataLoader(dg.data, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == '__main__':
    from config import Config

    Config["train_flag"] = True
    # dg = DataGenerator(Config["train_data_path"], Config)
    dataset = load_data_batch(Config["train_data_path"], Config)
    # print(len(dg))
    # print(dg[0])
    for index, dataset in enumerate(dataset):
        input_id1, input_id2, input_id3 = dataset
        print(input_id1)
        print(input_id2)
        print(input_id3)

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel

"""
建立网络模型结构
"""


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        output_size = config["output_size"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        self.use_bert = config["use_bert"]
        self.emb = nn.Embedding(vocab_size + 1, hidden_size, padding_idx=0)
        if model_type == 'rnn':
            self.encoder = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                                  batch_first=True)
        elif model_type == 'lstm':
            # 双向lstm，输出的是 hidden_size * 2(num_layers 要写2)
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        elif self.use_bert:
            self.encoder = BertModel.from_pretrained(config["bert_model_path"])
            # 需要使用预训练模型的hidden_size
            hidden_size = self.encoder.config.hidden_size
        elif model_type == 'cnn':
            self.encoder = CNN(config)
        elif model_type == "gated_cnn":
            self.encoder = GatedCNN(config)
        elif model_type == "bert_lstm":
            self.encoder = BertLSTM(config)
            # 需要使用预训练模型的hidden_size
            hidden_size = self.encoder.config.hidden_size

        self.classify = nn.Linear(hidden_size, output_size)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.functional.cross_entropy  # loss采用交叉熵损失

    def forward(self, x, y=None):
        if self.use_bert:
            # 输入x为[batch_size, seq_len]
            # bert返回的结果是 (sequence_output, pooler_output)
            # sequence_output:batch_size, max_len, hidden_size
            # pooler_output:batch_size, hidden_size
            x = self.encoder(x)[0]
        else:
            x = self.emb(x)
            x = self.encoder(x)
        # 判断x是否是tuple
        if isinstance(x, tuple):
            x = x[0]
        # 池化层
        if self.pooling_style == "max":
            # shape[1]代表列数，shape是行和列数构成的元组
            self.pooling_style = nn.MaxPool1d(x.shape[1])
        elif self.pooling_style == "avg":
            self.pooling_style = nn.AvgPool1d(x.shape[1])
        x = self.pooling_style(x.transpose(1, 2)).squeeze()

        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred, y.squeeze())
        else:
            return y_pred


# 定义孪生网络  （计算两个句子之间的相似度）
class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = TorchModel(config)
        # 使用的是cos计算
        # self.loss = nn.CosineEmbeddingLoss()
        # 使用triplet_loss
        self.triplet_loss = self.cosine_triplet_loss

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    # 3个样本  2个为一类 另一个一类 计算triplet loss
    def cosine_triplet_loss(self, a, p, n, margin=None):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin.squeeze()
        return torch.mean(diff[diff.gt(0)])  # greater than

    # 使用triplet_loss
    def forward(self, sentence1, sentence2=None, sentence3=None, margin=None):
        vector1 = self.sentence_encoder(sentence1)
        # 同时传入3 个样本
        if sentence2 is None:
            if sentence3 is None:
                return vector1
            # 计算余弦距离
            else:
                vector3 = self.sentence_encoder(sentence3)
                return self.cosine_distance(vector1, vector3)
        else:
            vector2 = self.sentence_encoder(sentence2)
            if sentence3 is None:
                return self.cosine_distance(vector1, vector2)
            else:
                vector3 = self.sentence_encoder(sentence3)
                return self.triplet_loss(vector1, vector2, vector3, margin)

    # CosineEmbeddingLoss
    # def forward(self,sentence1, sentence2=None, target=None):
    #     # 同时传入两个句子
    #     if sentence2 is not None:
    #         vector1 = self.sentence_encoder(sentence1)  # vec:(batch_size, hidden_size)
    #         vector2 = self.sentence_encoder(sentence2)
    #         # 如果有标签，则计算loss
    #         if target is not None:
    #             return self.loss(vector1, vector2, target.squeeze())
    #         # 如果无标签，计算余弦距离
    #         else:
    #             return self.cosine_distance(vector1, vector2)
    #     # 单独传入一个句子时，认为正在使用向量化能力
    #     else:
    #         return self.sentence_encoder(sentence1)


# 优化器的选择
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["lr"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


# 定义CNN模型
class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1) / 2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)

    def forward(self, x):  # x : (batch_size, max_len, embeding_size)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)


# 定义GatedCNN模型
class GatedCNN(nn.Module):
    def __init__(self, config):
        super(GatedCNN, self).__init__()
        self.cnn = CNN(config)
        self.gate = CNN(config)

    # 定义前向传播函数 比普通cnn多了一次sigmoid 然后互相卷积
    def forward(self, x):
        a = self.cnn(x)
        b = self.gate(x)
        b = torch.sigmoid(b)
        return torch.mul(a, b)


# 定义BERT-LSTM模型
class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_model_path"], return_dict=False)
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        x = self.bert(x)[0]
        x, _ = self.rnn(x)
        return x


if __name__ == "__main__":
    from config import Config

    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = SiameseNetwork(Config)
    s1 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    s2 = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    l = torch.LongTensor([[1], [0]])
    y = model(s1, s2, l)
    print(y)

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import SiameseNetwork, choose_optimizer
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
    #加载模型
    model = SiameseNetwork(config)
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
            input_id1, input_id2, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id1, input_id2, labels)
            train_loss.append(loss.item())
            # if index % int(len(train_data) / 2) == 0:
            #     logger.info("batch loss %f" % loss)
            loss.backward()
            optimizer.step()
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return

if __name__ == "__main__":
    main(Config)

"""
模型效果测试
"""
import torch
from loader import load_data_batch


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        # 选择验证集合
        config['train_flag'] = False
        self.valid_data = load_data_batch(config["valid_data_path"], config, shuffle=False)
        config['train_flag'] = True
        self.train_data = load_data_batch(config["train_data_path"], config)
        self.stats_dict = {"correct": 0, "wrong": 0}  # 用于存储测试结果

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空前一轮的测试结果
        self.model.eval()
        self.knwb_to_vector()
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                test_question_vectors = self.model(input_id)  # 不输入labels，使用模型当前参数进行预测
            self.write_stats(test_question_vectors, labels)
        self.show_stats()
        return

    def write_stats(self, test_question_vectors, labels):
        assert len(labels) == len(test_question_vectors)
        for test_question_vector, label in zip(test_question_vectors, labels):
            # 通过一次矩阵乘法，计算输入问题和知识库中所有问题的相似度
            # test_question_vector shape [vec_size]   knwb_vectors shape = [n, vec_size]
            res = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T)
            hit_index = int(torch.argmax(res.squeeze()))  # 命中问题标号
            hit_index = self.question_index_to_standard_question_index[hit_index]  # 转化成标准问编号
            if int(hit_index) == int(label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        for standard_question_index, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                # 记录问题编号到标准问题标号的映射，用来确认答案是否正确
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)
        with torch.no_grad():
            question_matrixs = torch.stack(self.question_ids, dim=0)
            if torch.cuda.is_available():
                question_matrixs = question_matrixs.cuda()
            self.knwb_vectors = self.model(question_matrixs)
            # 将所有向量都作归一化 v / |v|
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)




