# -*- coding: utf-8 -*-
import jieba
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
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        # self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    # 输入为问题字符编码
    def forward(self, x):
        # sentence_length = torch.sum(x.gt(0), dim=-1)
        x = self.embedding(x)
        # 使用lstm
        x, _ = self.layer(x)
        # 使用线性层
        # x = self.layer(x)
        # x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x


import torch


class SiameseNetwork(torch.nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = self.cosine_triplet_loss
        self.config = config
    # 计算余弦距离 1-cos(a,b)
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    def cosine_triplet_loss(self, a, p, n, margin=0.1):
        # 计算正样本与负样本的余弦距离
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)

        # margin默认为0.1，如果传入了margin值则使用传入的margin
        diff = ap - an + margin

        # 使用torch.maximum确保最小值为零，避免负值
        loss = torch.mean(torch.maximum(diff, torch.zeros_like(diff)))

        return loss

    # #sentence : (batch_size, max_length)
    # def forward(self, sentence1, sentence2=None, target=None):
    #     #同时传入两个句子
    #     if sentence2 is not None:
    #         vector1 = self.sentence_encoder(sentence1) #vec:(batch_size, hidden_size)
    #         vector2 = self.sentence_encoder(sentence2)
    #         #如果有标签，则计算loss
    #         if target is not None:
    #             return self.loss(vector1, vector2, target.squeeze())
    #         #如果无标签，计算余弦距离
    #         else:
    #             return self.cosine_distance(vector1, vector2)
    #     #单独传入一个句子时，认为正在使用向量化能力
    #     else:
    #         return self.sentence_encoder(sentence1)

    def forward(self, a, p=None, n=None):
        # 同时传入三个句子
        if p is not None:
            # Encode all three sentences
            vector_a = self.sentence_encoder(a)
            vector_p = self.sentence_encoder(p)
            if n is not None:
                vector_n = self.sentence_encoder(n)
                return self.loss(vector_a, vector_p, vector_n)
            else:
                return self.cosine_distance(vector_a, vector_p)
        else:
            return self.sentence_encoder(a)




def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config

    Config["vocab_size"] = 4662
    Config["max_length"] = 4
    model = SiameseNetwork(Config)
    s1 = torch.LongTensor([[1, 2, 3, 1], [2, 2, 0, 0]])
    s2 = torch.LongTensor([[0, 0, 0, 100], [2, 2, 0, 0]])
    l = torch.LongTensor([[1, 2, 3, 4], [2, 2, 3, 3]])
    # y = model(s1, s2, l)
    # x = model(s1)
    z = model(s1, s2, l)
    # print(x)
    # print(y)
    print(z)
    # print(model.state_dict())
