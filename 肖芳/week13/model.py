# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        # vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]

        bert = BertModel.from_pretrained(config["bert_path"],max_length=max_length)
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"]
        )

        self.layer = get_peft_model(bert, peft_config)

        self.classify = nn.Linear(hidden_size, class_num)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        x, _ = self.layer(x, return_dict=False)      #input shape:(batch_size, sen_len, 768)
        predict = self.classify(x) #ouput:(batch_size, sen_len, num_tags)
        
        if target is not None:
            predict_view = predict.view(-1, predict.shape[-1]) # (batch_size * sen_len, num_tags)
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