# -*- coding: utf-8 -*-
# @Time    : 2025/1/8 16:10
# @Author  : yeye
# @File    : model.py
# @Software: PyCharm
# @Desc    :

import torch.nn as nn
from transformers import BertModel
import torch
from torch.optim import Adam, SGD


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1  # 位置0是pad
        class_num = config["class_num"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        self.use_bert = False
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if model_type == "bert":
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"],
                                                     return_dict=False)  # 默认模型输出是个字典，不是字典则会返回元组
            hidden_size = self.encoder.config.hidden_size
        elif model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.classify = nn.Linear(hidden_size, class_num)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.functional.cross_entropy

    def forward(self, x, target=None):
        if self.use_bert:  # bert返回的结果是 (sequence_output, pooler_output)
            # sequence_output:batch_size, max_len, hidden_size
            # pooler_output:batch_size, hidden_size
            x = self.encoder(x)
        else:
            x = self.embedding(x)
            x = self.encoder(x)
        if isinstance(x, tuple):
            x = x[0]
        if self.pooling_style == 'avg':
            self.pooling_layer = nn.AvgPool1d(x.shape[1])
        else:
            self.pooling_layer = nn.MaxPool1d(x.shape[1])
        x = self.pooling_layer(x.transpose(1, 2)).squeeze()  # output shape:batch_size * hidden_size
        predict = self.classify(x)  # output shape:batch_size * class_num
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
if __name__ == "__main__":
    from config import Config

    Config["class_num"] = 3
    Config["vocab_size"] = 20
    Config["max_length"] = 5
    Config["model_type"] = "bert"
    model = BertModel.from_pretrained(Config["pretrain_model_path"], return_dict=False)
    x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    sequence_output, pooler_output = model(x)
    print(x[1], type(x[1]), len(x[1]))
    model = TorchModel(Config)
    label = torch.LongTensor([1, 2])
    print(model(x, label))
