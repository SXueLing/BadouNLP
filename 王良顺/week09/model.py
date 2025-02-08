# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        num_layers = config["num_layers"]
        self.class_num = config["class_num"]
        self.use_bert = config["use_bert"]
        # self.use_crf = config["use_crf"]
        self.bert = BertModel.from_pretrained(config["bert_path"])
        self.classify = nn.Linear(self.bert.config.hidden_size, self.class_num)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-100)  #loss采用交叉熵损失
        # self.crf_layer = CRF(self.class_num, batch_first=True)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        if self.use_bert:
            bert_output = self.bert(x)
            predict = self.classify(bert_output[0])
        else:
            x = self.embedding(x)
            x, _ = self.layer(x)
            predict = self.classify(x)

        if target is not None:
            return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
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
    model = TorchModel(Config)
