# -*- coding: utf-8 -*-
"""
建立网络模型结构
"""
import torch
import torch.nn as nn
from transformers import BertModel

from test_ai.homework.week07.config.running_config import Config


class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        x = self.bert(x)[0]
        x, _ = self.lstm(x)
        return x


if __name__ == "__main__":
    model = BertModel.from_pretrained(Config["pretrain_model_path"], return_dict=False)
    x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    sequence_output = model(x)
    print(x)
