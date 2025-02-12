# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel
from config import Config

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.class_num = config['class_num']
        self.bert = BertModel.from_pretrained(config['bert_path'])
        self.hidden_size = self.bert.config.hidden_size
        self.classify_layer = nn.Linear(self.hidden_size, self.class_num) # 接受bert的输出然后分类
        self.crf_layer = CRF(self.class_num, batch_first = True)
        self.use_crf = config['use_crf']

        self.loss_fn = nn.CrossEntropyLoss(ignore_index = -1) # 计算loss时忽视label是-1，也就是padding的位置，即padding不参与训练

    def forward(self, input_ids, attention_mask, label_ids = None):
        '''
        input_ids : (batch_size, max_len) -> 词在bert词表中的序号
        attention_mask: (batch_size, max_len) -> 关注掩码（1:有效, 0:padding）
        labels: (batch_size, max_len) -> 真实标签（可选）
        return: loss（训练时）或预测结果（推理时）
        '''

        bert_output = self.bert(input_ids, attention_mask = attention_mask)
        '''
        1. last_hidden_state
        形状: (batch_size, max_length, hidden_size)
        作用: 每个 token 的表示向量，编码了它的上下文信息。
        适用于: 序列标注任务（如 NER），因为它提供 每个 token 的特征。

        2. pooler_output
        形状: (batch_size, hidden_size)
        作用: CLS 位置的隐藏状态，经过 tanh 变换后输出一个句子级别的向量。
        适用于: 分类任务（如情感分析），因为它提供 整个序列 的表示。
        '''
        sequence_output = bert_output.last_hidden_state
        predict = self.classify_layer(sequence_output) # (batch_size, max_length, class_num)

        if label_ids is not None: # 训练模式，计算loss
            if self.use_crf:
                mask = label_ids.gt(-1) 
                return - self.crf_layer(predict, label_ids, mask, reduction="mean")
            else:
                # 预测形状(batch_size*max_len, class_num), 真实标签形状(batch_size*max_len,)
                loss = self.loss_fn(predict.view(-1, predict.shape[-1]), label_ids.view(-1))
                return loss
        
        else: # 预测模式，计算预测的标签
            if self.use_crf:
                return self.crf_layer.decode(predict) # 维比特解码
            else:
                return predict.argmax(-1)


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

    input_ids = torch.tensor([[101, 795, 4896, 829, 6930, 672, 2400, 6630, 118, 801, 2219, 1763,
                               5441, 5992, 2200, 5967, 1502, 2936, 4347, 5315, 1745, 2158, 4639, 7310,
                               7580, 512, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)  # 输入的词索引
    attention_mask = torch.tensor([[1] * 100], dtype=torch.long)  # 输入的注意力掩码
    label_ids = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0]], dtype=torch.long)  # 真实标签（包括padding）

    loss = model(input_ids=input_ids, attention_mask=attention_mask, label_ids=label_ids)
    print(loss)

    predict = model(input_ids=input_ids, attention_mask=attention_mask)
    print(predict)
