# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
"""
建立网络模型结构
"""

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        model_type = config["model_type"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if model_type == "lstm":
            self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        elif model_type == "gru":
            self.layer = nn.GRU(hidden_size, hidden_size, batch_first=True)
        elif model_type == "bilstm":
            self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
            hidden_size = hidden_size * 2
        elif model_type == "rnn":
            self.layer = nn.RNN(hidden_size, hidden_size, batch_first=True)
        elif model_type == "cnn":
            self.layer = CNN(config)
        elif model_type == "gated_cnn":
            self.layer = GatedCNN(config)
        elif model_type == "dnn":
            self.layer = nn.Sequential(
                nn.Linear(hidden_size, 2*hidden_size),
                nn.ReLU(),
                nn.Linear(2*hidden_size, hidden_size)
            )
            # self.layer = nn.Linear(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    #输入为问题字符编码
    def forward(self, x):
        x = self.embedding(x)
        x = self.layer(x)
        #使用rnn类的模型
        if isinstance(x, tuple):
            x = x[0]
        # x = self.linear(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        hidden_size = config["hidden_size"]
        kernel_size = 3
        pad = int((kernel_size - 1)/2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)

    def forward(self, x): #x : (batch_size, max_len, embeding_size)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)

class GatedCNN(nn.Module):
    def __init__(self, config):
        super(GatedCNN, self).__init__()
        self.cnn = CNN(config)
        self.gate = CNN(config)

    def forward(self, x):
        a = self.cnn(x)
        b = self.gate(x)
        b = torch.sigmoid(b)
        return torch.mul(a, b)

class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = self.cosine_triplet_loss  #Triplet loss
        # self.loss = nn.CosineEmbeddingLoss()

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    def cosine_triplet_loss(self, a, p, n, margin=None):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.5
        else:
            diff = ap - an + margin
        return torch.mean(diff[diff.gt(0)]) #greater than

    #sentence : (batch_size, max_length)
    def forward(self, sentence1, sentence2=None, sentence3=None, Margin=None):
        #同时传入三个句子
        if sentence2 is not None and sentence3 is not None:
            vector1 = self.sentence_encoder(sentence1) #vec:(batch_size, hidden_size)
            vector2 = self.sentence_encoder(sentence2)
            vector3 = self.sentence_encoder(sentence3)
            return self.loss(vector1, vector2, vector3, Margin)
        #单独传入一个句子时，认为正在使用向量化能力
        else:
            return self.sentence_encoder(sentence1)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = SiameseNetwork(Config)
    s1 = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    s2 = torch.LongTensor([[1,2,3,4], [3,2,3,4]])
    l = torch.LongTensor([[1],[0]])
    y = model(s1, s2, l)
    print(y)
    # print(model.state_dict())