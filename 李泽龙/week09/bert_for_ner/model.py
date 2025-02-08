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
        self.use_bert = config.get("use_bert", False)
        self.use_crf = config["use_crf"]
        self.max_length = config["max_length"]
        self.class_num = config["class_num"]
        
        if self.use_bert:
            self.bert = BertModel.from_pretrained(config["bert_path"])
            self.classify = nn.Linear(self.bert.config.hidden_size, self.class_num)
        else:
            hidden_size = config["hidden_size"]
            vocab_size = config["vocab_size"] + 1
            self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
            self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=config["num_layers"])
            self.classify = nn.Linear(hidden_size * 2, self.class_num)
        
        self.crf_layer = CRF(self.class_num, batch_first=True)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-500)  # 修改ignore_index为-500

    def forward(self, x, target=None, attention_mask=None):
        if self.use_bert:
            bert_output = self.bert(x, attention_mask=attention_mask)
            sequence_output = bert_output.last_hidden_state
            predict = self.classify(sequence_output)
        else:
            x = self.embedding(x)
            x, _ = self.layer(x)
            predict = self.classify(x)
        
        if target is not None:
            if self.use_crf:
                mask = target.gt(-500)  # 忽略-500的标签
                return -self.crf_layer(predict, target, mask, reduction="mean")
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


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)
