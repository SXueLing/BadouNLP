# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel, BertPreTrainedModel
import torch.nn.functional as F

"""
建立网络模型结构
"""

class BertModelWithCRF(nn.Module):
    def __init__(self, config):
        super(BertModelWithCRF, self).__init__()
        max_length = config["max_length"]
        class_num = config["class_num"]

        self.bert = BertModel.from_pretrained(config["bert_path"], return_dict=False)
        self.hidden_size = self.bert.config.hidden_size

        self.classify = nn.Linear(self.hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  # 忽略标签为-1的样本
    
    def forward(self, input_ids, attention_mask, target=None):
        input_ids, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask) # (batch_size, sen_len, hidden_size)
        predict = self.classify(input_ids)  #  (batch_size, seq_len, hidden_size) -> (batch_size * seq_len, class_num)
        
        if target is not None:
            if self.use_crf:
                # mask = target.gt(-1) # 不使用attention_mask
                loss = - self.crf_layer(predict, target, mask=attention_mask.bool(), reduction="mean")
                return loss
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict, mask=attention_mask.bool())
            else:
                _, predict = torch.max(predict, dim=-1)
                return predict
            
class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.classify = nn.Linear(hidden_size * 2, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        x = self.embedding(x)  #input shape:(batch_size, sen_len)
        x, _ = self.layer(x)      #input shape:(batch_size, sen_len, input_dim)
        predict = self.classify(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1) 
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                # number = batch_size * seq_len
                #(number, class_num), (number)
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:   
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict

class BERTLSTMCRF(nn.Module):
    def __init__(self, config):
        super(BERTLSTMCRF, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_path"])
        self.lstm = nn.LSTM(
            input_size = self.bert.config.hidden_size,
            hidden_size = config["hidden_size"],
            num_layers = config["num_layers"],
            bidirectional = config["bidirectional"],
            batch_first = True
        )
        self.dropout = nn.Dropout(config["dropout"])
        # 计算LSTM输出维度
        lstm_outdim = config["hidden_size"] * (2 if config["bidirectional"] else 1)
        self.fc = nn.Linear(lstm_outdim, config["class_num"])
        self.crf_layer = CRF(config["class_num"], batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    def forward(self, input_ids, attention_mask, target=None):
        # Bert编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs.last_hidden_state # [batch_size, seq_len, hidden_dim]
        # LSTM处理
        lstm_out, _ = self.lstm(outputs)
        lstm_out = self.dropout(lstm_out)
        # 全连接层
        predict = self.fc(lstm_out)

        if target is not None:
            if self.use_crf:
                # CRF模式：返回标量损失值
                loss = -self.crf_layer(predict, target, mask=attention_mask.bool())
                return loss  # 确保返回的是标量
            else:
                # 非CRF模式：计算CrossEntropyLoss
                loss = self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
                return loss  # 确保返回的是标量
        else:
            if self.use_crf:
                # CRF解码
                best_paths = self.crf_layer.decode(predict, mask=attention_mask.bool())
                return best_paths
            else:
                # 非CRF模式：直接取最大概率标签
                _, best_paths = torch.max(predict, dim=-1)
                return best_paths

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
    
