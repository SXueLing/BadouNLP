# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel


class TorchModel(nn.Module):
    """
    建立网络模型结构
    """

    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1  # todo 这里为什么加1？
        class_num = config["class_num"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        self.use_bert = False
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if model_type == "fast_text":
            self.encoder = lambda x: x
        elif model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "gru":
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "bert":
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        self.classify = nn.Linear(hidden_size, class_num)
        self.pooling_style = config["pooling_style"]
        # loss 采用交叉熵计算损失
        self.loss = nn.functional.cross_entropy

    def forward(self, x, target=None):
        if self.use_bert:
            x = self.encoder(x)
        else:
            x = self.embedding(x)
            x = self.encoder(x)

        if isinstance(x, tuple):
            x = x[0]

        if self.pooling_style == "max":
            self.pooling_layer = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])
        x = self.pooling_layer(x.transpose(1, 2)).squeeze()
        predict = self.classify(x)
        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
