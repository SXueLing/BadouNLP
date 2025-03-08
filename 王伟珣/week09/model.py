# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel
from torchcrf import CRF


class NerModel(nn.Module):
    def __init__(self, config):
        super(NerModel, self).__init__()
        num_layers = config['num_layers']
        n_classes = config['n_classes']
        bert_path = config['bert_path']
        hidden_size = config['hidden_size']

        self.bert = BertModel.from_pretrained(
            bert_path, return_dict=False, ignore_mismatched_sizes=True
        )
        bert_hidden_size = self.bert.config.hidden_size
        
        self.lstm = nn.LSTM(
            bert_hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers
        )
        self.linear = nn.Linear(hidden_size * 2, n_classes)
        
        self.use_crf = config['use_crf']
        if self.use_crf:
            self.crf_layer = CRF(n_classes, batch_first=True)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        return

    def forward(self, inputs, target=None):
        x = self.bert(inputs)[0]

        x, _ = self.lstm(x)
    
        predict = self.linear(x)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1) 
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
