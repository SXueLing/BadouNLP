# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig
from torch.nn import TripletMarginLoss  # 引入TripletMarginLoss

"""
建立网络模型结构
"""

class GetFirst(nn.Module):
    def __init__(self):
        super(GetFirst, self).__init__()

    def forward(self, x):
        return x[0]

class SentenceMatchNetwork(nn.Module):
    def __init__(self, config):
        super(SentenceMatchNetwork, self).__init__()
        # 可以用bert，参考下面
        # pretrain_model_path = config["pretrain_model_path"]
        # self.bert_encoder = BertModel.from_pretrained(pretrain_model_path)

        # 常规的embedding + layer
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
        # self.classify_layer = nn.Linear(hidden_size, 2)  # 移除分类层，因为Triplet Loss不需要
        self.loss_fn = TripletMarginLoss(margin=1.0)  # 定义Triplet Loss

    # 同时传入三个句子的拼接编码（anchor, positive, negative）
    def forward(self, anchor_ids, positive_ids, negative_ids):
        # x = self.bert_encoder(input_ids)[1]
        # input_ids = batch_size, max_length
        anchor_x = self.embedding(anchor_ids)  # anchor_x:batch_size, max_length, embedding_size
        anchor_x = self.encoder(anchor_x)
        anchor_x = nn.MaxPool1d(anchor_x.shape[1])(anchor_x.transpose(1, 2)).squeeze()  # anchor_x: batch_size, hidden_size

        positive_x = self.embedding(positive_ids)  # positive_x:batch_size, max_length, embedding_size
        positive_x = self.encoder(positive_x)
        positive_x = nn.MaxPool1d(positive_x.shape[1])(positive_x.transpose(1, 2)).squeeze()  # positive_x: batch_size, hidden_size

        negative_x = self.embedding(negative_ids)  # negative_x:batch_size, max_length, embedding_size
        negative_x = self.encoder(negative_x)
        negative_x = nn.MaxPool1d(negative_x.shape[1])(negative_x.transpose(1, 2)).squeeze()  # negative_x: batch_size, hidden_size

        loss = self.loss_fn(anchor_x, positive_x, negative_x)  # 计算Triplet Loss
        return loss


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
    s1 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    s2 = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    s3 = torch.LongTensor([[4, 3, 2, 1], [0, 0, 2, 3]])
    loss = model(s1, s2, s3)
    print(loss)