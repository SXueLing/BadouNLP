import torch
import torch.nn as nn
from config_file import Config
from transformers import BertModel


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.bert = BertModel.from_pretrained(Config["pretrain_model_path"],
                                              return_dict=False,
                                              attn_implementation='eager')
        self.hidden_size = Config["hidden_size"]
        self.rnn = nn.LSTM(input_size=self.bert.config.hidden_size,
                           hidden_size=self.hidden_size,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=True)
        self.classify = nn.Sequential(
            #nn.Linear(self.bert.config.hidden_size, self.hidden_size),
            #nn.GELU(),
            #nn.Dropout(0.3),
            nn.Linear(self.bert.config.hidden_size, Config["num_labels"])
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)


    def forward(self, input_ids, attention_mask=None, target=None):
        if attention_mask is None:
            x, _ = self.bert(input_ids)
        else:
            x, _ = self.bert(input_ids, attention_mask=attention_mask)
        #x, _ = self.rnn(x)
        predict = self.classify(x)
        if target is not None:
            loss = self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
            return loss
        else:
            y_p = nn.functional.softmax(predict, dim=-1)
            return y_p

def choose_optimizer(model):
    if Config["optim_type"] == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=Config["lr"])
    else:
        optim = torch.optim.SGD(model.parameters(), lr=Config["lr"])
    return optim
