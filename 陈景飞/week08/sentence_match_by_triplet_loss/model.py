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
        hidden_size = config["hidden_size"] # 128
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"] #30
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    #输入为问题字符编码
    def forward(self, x):
        x = self.embedding(x)
        #使用lstm
        # x, _ = self.lstm(x)
        #使用线性层
        x = self.layer(x)
        x = nn.functional.max_pool1d(x.transpose(0,1), x.shape[0]).squeeze()
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = nn.CosineEmbeddingLoss()

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
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin.squeeze()
        return diff if diff.gt(0) else torch.tensor(0)

    #计算一个batch的
    def batch_cosine_triplet_loss(self, batch_size, a, p, n):
        result = []
        for i in range(batch_size):
            v1 = self.sentence_encoder(a[i])
            v2 = self.sentence_encoder(p[i])
            v3 = self.sentence_encoder(n[i])
            result.append(self.cosine_triplet_loss(v1, v2, v3))
        result = sum(result) / batch_size
        return result.requires_grad_(True)

    # 预测时候使用
    def batch_distance_for_predict(self, batch_size, a, p, n):
        result = []
        for i in range(batch_size):
            v1 = self.sentence_encoder(a[i])
            v2 = self.sentence_encoder(p[i])
            v3 = self.sentence_encoder(n[i])
            v1_to_v2 = self.cosine_distance(v1, v2)
            v1_to_v3 = self.cosine_distance(v1, v3)
            if v1_to_v3.gt(v1_to_v2):
                result.append(torch.LongTensor([1]))
            else:
                result.append(torch.LongTensor([0]))
        return result




    #sentence : (batch_size, max_length)
    def forward(self, a, p=None, n=None, target=None):
        batch_size, _ = a.shape
        if target is not None:
            return self.batch_cosine_triplet_loss(batch_size, a, p, n)
        else:
            return self.batch_distance_for_predict(batch_size, a, p, n)


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