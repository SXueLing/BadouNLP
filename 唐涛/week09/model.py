# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from transformers import BertModel
from transformers import BertConfig
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
"""
建立网络模型结构
实现sentence level的序列标注
每次只跑一个段落
"""


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.bert_like = BertModel.from_pretrained(config["pretrain_model_path"])
        self.dropout = nn.Dropout(self.bert_like.config.hidden_dropout_prob)
        self.num_labels = config["num_labels"]
        if config["recurrent"] == "lstm":
            self.recurrent_layer = nn.LSTM(self.bert_like.config.hidden_size,
                                      self.bert_like.config.hidden_size // 2,
                                      batch_first=True,
                                      bidirectional=True,
                                      num_layers=1
                                      )
        elif config["recurrent"] == "gru":
            self.recurrent_layer = nn.GRU(self.bert_like.config.hidden_size,
                                      self.bert_like.config.hidden_size // 2,
                                      batch_first=True,
                                      bidirectional=True,
                                      num_layers=1
                                      )
        else:
            assert False

        self.classifier = nn.Linear(self.bert_like.config.hidden_size, config["num_labels"])

        self.classifier1 = nn.Linear(self.bert_like.config.hidden_size, config["num_labels"])
        self.classifier2 = nn.Linear(self.bert_like.config.hidden_size, config["num_labels"])
        self.classifier3 = nn.Linear(self.bert_like.config.hidden_size, config["num_labels"])
        self.classifier4 = nn.Linear(self.bert_like.config.hidden_size, config["num_labels"])

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, input_ids=None,
                      attention_mask=None,
                      labels=None):
        #(num_sentence, sentence_length)
        outputs = self.bert_like(
            input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = outputs[1]
        #(num_sentence, vector_size=768)
        pooled_output = self.dropout(pooled_output)
        #(num_sentence, vector_size=768) -> (1, num_sentence, vector_size)
        recurrent_output, _ = self.recurrent_layer(pooled_output.unsqueeze(0))
        #(1, num_sentence, LSTM_HIDDEN_SIZE * 2)
        output = self.classifier(recurrent_output.squeeze(0))

        if labels is not None:
            loss_func = CrossEntropyLoss()
            loss = loss_func(output.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return output

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
    elif optimizer == "adamw":
        return AdamW(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    Config["optimizer"] = "adamw"
    model = TorchModel(Config)
    input_ids = torch.LongTensor([[0,1,2,3,4,100,6,7,8], [0,4,3,2,1,100,8,7,6]])
    labels = torch.LongTensor([[1], [0]])
    print(model(input_ids, labels=labels))
