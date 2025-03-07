# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel
import loralib as lora
"""
建立网络模型结构
"""

# 修改后的model.py

class TorchModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config["bert_path"], return_dict=False)
        
        # 注入LoRA到BERT的注意力机制
        for layer_idx in range(len(self.bert.encoder.layer)):
            # 修改query
            self.bert.encoder.layer[layer_idx].attention.self.query = lora.Linear(
                self.bert.config.hidden_size,
                self.bert.config.hidden_size,
                r=config.get("lora_r", 8),
                lora_alpha=config.get("lora_alpha", 16)
            )

            self.bert.encoder.layer[layer_idx].attention.self.value = lora.Linear(
                self.bert.config.hidden_size,
                self.bert.config.hidden_size,
                r=config.get("lora_r", 8),
                lora_alpha=config.get("lora_alpha", 16)
            )

        # 分类层（保持原始结构）
        self.classify = nn.Linear(self.bert.config.hidden_size, config["class_num"])
        self.crf_layer = CRF(config["class_num"], batch_first=True) if config["use_crf"] else None
        self.use_crf = config["use_crf"]
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

        # 参数冻结策略
        lora.mark_only_lora_as_trainable(self.bert)  # 冻结BERT非LoRA参数
        if self.crf_layer is not None:
            self.crf_layer.requires_grad_(True)  # 保持CRF层可训练


    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        # x = self.embedding(x)  #input shape:(batch_size, sen_len)
        # x, _ = self.layer(x)      #input shape:(batch_size, sen_len, input_dim)
        # predict = self.classify(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

        x, _ = self.bert(x)
        predict = self.classify(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)



        if target is not None:
            if self.use_crf:
                mask = target.gt(-1) 
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                #(number, class_num), (number)
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
