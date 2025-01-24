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
        self.bert = BertModel.from_pretrained(config['bert_path'])
        hidden_size = self.bert.config.hidden_size
        class_num = config["class_num"]
        self.classify = nn.Linear(hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-100)  # BERT常用-100作为忽略的标签
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, target=None):
        # input_ids: (batch_size, seq_len)
        # attention_mask: (batch_size, seq_len)
        # target: (batch_size, seq_len)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        sequence_output = self.dropout(sequence_output)
        predict = self.classify(sequence_output)  # (batch_size, seq_len, num_tags)

        if target is not None:
            if self.use_crf:
                # 使用attention_mask作为CRF的mask
                return - self.crf_layer(predict, target, attention_mask.bool(), reduction="mean")
            else:
                # 只计算非padding位置的loss
                active_loss = attention_mask.view(-1) == 1
                active_logits = predict.view(-1, predict.shape[-1])[active_loss]
                active_labels = target.view(-1)[active_loss]
                return self.loss(active_logits, active_labels)
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