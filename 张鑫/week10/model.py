# -*- coding: utf-8 -*-

"""
网络模型结构（使用bert）
"""
import torch.nn
from torch import nn
from transformers import BertModel


class LanguageModel(torch.nn.Module):

    def __init__(self, config):
        super(LanguageModel, self).__init__()
        self.bert_layer = BertModel.from_pretrained(config['bert_path'], return_dict=False)
        self.config = config
        self.hidden_size = self.bert_layer.config.hidden_size
        self.vocab_size = self.bert_layer.config.vocab_size
        self.classify = nn.Linear(self.hidden_size, self.vocab_size)
        self.loss = nn.functional.cross_entropy

    def forward(self, input_data, target=None):
        if target is not None:
            # 训练模式，需要增加mask
            mask = torch.tril(torch.ones((input_data.shape[0], input_data.shape[1], input_data.shape[1])))
            if torch.cuda.is_available():
                mask = mask.cuda()
            predict = self.bert_layer(input_data, attention_mask=mask)[0]  # output shape: (batch_size, max_length, hidden_size)
            predict = self.classify(predict)  # output shape: (batch_size, max_length, vocab_size)
            # 计算loss: (batch_size * max_length, vocab_size), (batch_size * max_length)
            return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            # 生成模式
            predict = self.bert_layer(input_data)[0]
            predict = self.classify(predict)
            return torch.softmax(predict, dim=-1)
