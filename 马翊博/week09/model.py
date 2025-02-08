# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel
from TorchCRF import CRF

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        num_layers = config["num_layers"]
        self.bert = BertModel.from_pretrained(config["bert_path"])
        if num_layers is not None:
            # 保留前num_layers层，移除多余的层
            self.bert.encoder.layer = self.bert.encoder.layer[:num_layers]
        class_num = config["class_num"]
        hidden_size = self.bert.config.hidden_size
        self.classify = nn.Linear(hidden_size, class_num)
        self.crf_layer = CRF(class_num)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  # loss采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, input_ids, attention_mask, target=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        predict = self.classify(sequence_output)  # output: (batch_size, seq_len, num_tags)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
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



