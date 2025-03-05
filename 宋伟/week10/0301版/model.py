# -*- coding: utf-8 -*-
# @Date    :2025-03-01 23:49:33
# @Author  :SongWei (songweiof@gmail.com)
# @Link    :https://github.com/Finder12345
# @Version :python 3.11
# @Software:Sublime Text


import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel


class LanguageModel(nn.Module):
    """docstring for LanguageModel"""

    def __init__(self, config):
        super(LanguageModel, self).__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.vocab_size = config['vocab_size']

        self.bert = BertModel.from_pretrained(config['bert_path'],
                                              return_dict=False,
                                              attn_implementation="eager")
        self.classify = nn.Linear(
            self.bert.config.hidden_size, self.vocab_size)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        if y is not None:
            #训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1]))).to(x.device)
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            #预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)


    def create_causal_mask(self, seq_len):
        # 根据序列长度进行掩码处理
        mask = torch.tril(torch.ones(seq_len, seq_len))  # 生成一个下三角矩阵
        return mask.unsqueeze(0)  # 适配输入序列的格式,1,seq,seq


def choose_optimizer(model: nn.Module, config):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == 'adam':
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == '__main__':
    from config import Config
    from loader import data_load
    dataloader = data_load(Config)
    model = LanguageModel(Config)
    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            inputs, labels = batch_data['inputs'], batch_data['labels']
            loss = model(inputs, labels)
            print(loss)
            break
