# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchcrf import CRF

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_path"])
        self.class_num = config["class_num"]
        self.classify = nn.Linear(768, self.class_num)  # BERT的输出维度是768
        self.crf_layer = CRF(self.class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, target=None):
        # BERT的输入是input_ids和attention_mask
        attention_mask = (x != 0).long()  # 生成attention mask
        outputs = self.bert(x, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # 获取BERT的最后一层输出
        predict = self.classify(sequence_output)  # 分类层

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return - self.crf_layer(predict, target, mask, reduction="mean")
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