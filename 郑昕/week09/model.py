# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel, BertTokenizer

"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        
        # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        
        # Load pre-trained BERT model and tokenizer
        self.bert = BertModel.from_pretrained('bert-base-chinese')  
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
        # Freeze BERT layers if needed
        for param in self.bert.parameters():
            param.requires_grad = False
            
        
        self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.classify = nn.Linear(hidden_size * 2, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        # tokenize input text and convert to tensors
        inputs = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        input_ids = inputs['input_ids']  
        attention_mask = inputs['attention_mask']
        
        # Pass through BERT model
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state # shape: (batch_size, seq_len, hidden_size)
        
        
        # x = self.embedding(x)  #input shape:(batch_size, sen_len)
        
        # Additional layers after BERT
        x, _ = self.layer(x)      #input shape:(batch_size, sen_len, input_dim)
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