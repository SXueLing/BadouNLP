# -*- coding: utf-8 -*-

"""
网络模型结构（使用bert）
"""
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF

from transformers import BertModel


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        class_num = config['class_num']
        self.encoder = BertModel.from_pretrained(config['bert_path'], return_dict=False)
        hidden_size = self.encoder.config.hidden_size
        self.classify = nn.Linear(hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config['use_crf']
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, target=None):
        # batch_size * max_length ->
        # (sentence_output[batch_size, max_length, hidden_size], pooler_output[batch_size, hidden_size])
        x = self.encoder(x)[0]
        # batch_size, max_length, class_num
        predict = self.classify(x)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                # predict shape: (batch_size, max_length, class_num) -> (batch_size*max_length, class_num)
                # !: predict.shape[-1]=9, 是为了获取class_num, predict.view(-1, 9)是为了把3维向量转化为2维，
                # 即得到(batch_size*max_length, class_num)，将batch_size中的时间步都合并了
                # target shape: (batch_size, max_length)
                # (max_length, class_num)(max_length)
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
