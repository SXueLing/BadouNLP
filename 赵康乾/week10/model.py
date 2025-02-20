# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertTokenizer, BertModel
from config import Config

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.config = config

        self.bert = BertModel.from_pretrained(config['bert_path'])
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_path'])
        self.vocab_size = self.tokenizer.vocab_size
        self.classify = nn.Linear(self.bert.config.hidden_size, self.vocab_size)
        # self.softmax = nn.Softmax(dim = -1)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, id_input, attention_mask, id_output = None):
        #print(f'attention_mask shape: {attention_mask.shape}')
        #print(f'input shape: {id_input.shape}' )
        bert_output = self.bert(id_input, attention_mask = attention_mask)['last_hidden_state'] # [batch_size, seq_len, hidden_size]
        #print(f'bert_output shape: {bert_output.shape}' )
        predict = self.classify(bert_output) # [batch_size, seq_len, vocab_size]
        # predict = self.softmax(predict)
        #print(f'predict shape: {predict.shape}')

        if id_output is not None:
            return self.loss(predict.view(-1, predict.shape[-1]), id_output.view(-1)) # [batch_size*seq_len, vocab_size], [batch_size, seq_len]
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
    input_ids = torch.tensor([[
        1920, 2584, 6887, 8038,  100, 2207, 6777, 8024,  872, 2397, 
        *[0] * 90
        ]])

    target_ids = torch.tensor([[
        2584, 6887, 8038,  100, 2207, 6777, 8024,  872, 2397,  784, 
        *[0] * 90
        ]])

    max_length = 100
    window_size = 10
    mask = torch.tril(torch.ones(1, window_size, window_size))
    attention_mask = torch.zeros(1, max_length, max_length)
    attention_mask[:, :window_size, :window_size] = mask

    predict = model(id_input = input_ids, attention_mask = attention_mask)
    loss = model(id_input = input_ids, attention_mask = attention_mask, id_output = target_ids)

    print(f'predict is {predict}')
    print(f'loss is {loss}')
    
