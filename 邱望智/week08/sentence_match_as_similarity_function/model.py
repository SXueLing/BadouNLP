# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig

"""
建立网络模型结构
"""

class GetFirst(nn.Module):
    def __init__(self):
        super(GetFirst, self).__init__()

    def forward(self, x):
        return x[0]

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        # 20000应为词表大小，这里借用bert的词表，没有用它精确的数字，因为里面有很多无用词，舍弃一部分，不影响效果
        self.embedding = nn.Embedding(20000, hidden_size)
        # 一种多层按顺序执行的写法，具体的层可以换
        # unidirection:batch_size, max_len, hidden_size
        # bidirection:batch_size, max_len, hidden_size * 2
        self.encoder = nn.Sequential(nn.LSTM(hidden_size, hidden_size, bidirectional=True, batch_first=True),
                                     GetFirst(),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size * 2, hidden_size),  # batch_size, max_len, hidden_size
                                     nn.ReLU(),
                                     )

    #输入为问题字符编码
    def forward(self, x):
        x = self.embedding(x)  # x:batch_size, max_length, embedding_size
        x = self.encoder(x)  #
        return nn.MaxPool1d(x.shape[1])(x.transpose(1, 2)).squeeze()

class SentenceMatchNetwork(nn.Module):
    def __init__(self, config):
        super(SentenceMatchNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = nn.TripletMarginLoss(margin=1.0, p=2)

    # 同时传入两个句子的拼接编码
    # 输出一个相似度预测，不匹配的概率
    def forward(self, x, train=True):
        if train:
            input_ids1 = self.sentence_encoder(x[0])
            input_ids2 = self.sentence_encoder(x[1])
            input_ids3 = self.sentence_encoder(x[2])
            return self.loss(input_ids1, input_ids2, input_ids3)
        #预测相似度
        else:
            return self.sentence_encoder(x) #如果改为x[:,0]则是两句话不匹配的概率

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
    model = SentenceMatchNetwork(Config)
    s1 = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    s2 = torch.LongTensor([[1,2,3,4], [3,2,3,4]])
    l = torch.LongTensor([[1],[0]])
    # y = model(s1, s2, l)
    # print(y)
    # print(model.state_dict())
