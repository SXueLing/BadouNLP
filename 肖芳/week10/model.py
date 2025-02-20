# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel, BertConfig
from CustomBertModel import CustomBertModel
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        max_length = config["max_length"]
        bertConfig = BertConfig.from_pretrained(config["bert_path"])
        self.bert = CustomBertModel.from_pretrained(config["bert_path"],max_length=max_length, return_dict=False)
        self.classify = nn.Linear(hidden_size, bertConfig.vocab_size)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        x, _ = self.bert(x)      #input shape:(batch_size, sen_len, 768)
        predict = self.classify(x) # (batch_size, sen_len, vocab_size)
        
        if target is not None:
            predict_view = predict.view(-1, predict.shape[-1])
            target_view = target.view(-1)
            return self.loss(predict_view, target_view)
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