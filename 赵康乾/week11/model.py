# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertTokenizer, BertModel, BertConfig
from config import Config

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.config = config

        bert_config = BertConfig.from_pretrained(config['bert_path'])
        bert_config.num_hidden_layers = 1
        self.bert_layer = BertModel.from_pretrained(config['bert_path'], config=bert_config)
        self.hidden_size = self.bert_layer.config.hidden_size

        self.tokenizer = BertTokenizer.from_pretrained(config['bert_path'])
        self.vocab_size = self.tokenizer.vocab_size

        self.classify_layer = nn.Linear(self.hidden_size, self.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, mask, question_and_answer, answer = None):
        # question_and_answer [batch_size, max_length]
        # answer [batch_size, max_length]

        # answer_pred [batch_size, max_length, hidden_size]
        answer_pred = self.bert_layer(question_and_answer, attention_mask = mask)['last_hidden_state']
        # answer_pred [batch_size, max_length, vocab_size]
        answer_pred = self.classify_layer(answer_pred)

        if answer is not None:
            # [batch_size * max_length, hidden_size] , [batch_size * max_length, ]
            loss = self.loss_fn(answer_pred.view(-1, answer_pred.shape[-1]), answer.view(-1))
            return loss
        else:
            return torch.softmax(answer_pred, dim = -1)
        
def choose_optimizer(config, model):
    optimizer = config['optimizer']
    learning_rate = config['learning_rate']
    if optimizer == 'adam':
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'SGD':
        return SGD(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)
    input_ids = torch.tensor([[
        101, 2584, 6887, 8038, 102, 2207, 6777, 8024, 872, 102, 
        *[0] * 190
        ]])

    target_ids = torch.tensor([[
        -100, -100, -100,  -100, 2207, 6777, 8024, 872, 102,  -100, 
        *[-100] * 190
        ]])

    max_length = 200
    title_length = 3
    l_mask = torch.ones(1, max_length, (title_length + 1))
    r_mask = torch.zeros(1, max_length, (max_length - title_length - 1))
    tril_mask = torch.tril(torch.ones(1, (max_length - title_length - 1), (max_length - title_length - 1)))
    r_mask[: ,(title_length + 1):, :(max_length - title_length - 1)] = tril_mask
    attention_mask = torch.cat([l_mask, r_mask], dim=-1)

    predict = model(mask = attention_mask, question_and_answer = input_ids)
    loss = model(mask = attention_mask, question_and_answer = input_ids, answer = target_ids)

    print(f'predict is {predict}')
    print(f'loss is {loss}')
