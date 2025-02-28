import torch
from transformers import BertModel,BertConfig
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        if self.config["model_type"] == "bert":
            self._init_bert()
        else:
            self._init_vanilla()
        self.classify = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self. config["vocab_size"])
        )
        self.activation = nn.functional.softmax
        self.loss =  nn.functional.cross_entropy
        self.embedding = nn.Embedding(config["vocab_size"], config["hidden_size"])

    def _init_bert(self):
        self.bert = BertModel.from_pretrained(self.config["bert_path"], return_dict=False, attn_implementation='eager')
        self.hidden_size = self.bert.config.hidden_size

    def _init_vanilla(self):
        self.hidden_size = self.config["hidden_size"]
        self.embedding = nn.Embedding(
            self.config["vocab_size"],
            self.hidden_size
        )
        self.rnn = {
            "rnn":nn.RNN,
            "gru":nn.GRU,
            "lstm":nn.LSTM
        }[self.config["model_type"]]
        self.encoder = self.rnn(
            self.hidden_size,
            self.hidden_size,
            batch_first = True,
            num_layers = self.config["num_layers"]
        )

    def forward(self,x, y=None):
        if hasattr(self, "bert"):
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            x,_ = self.bert(x,attention_mask=mask)
        else:
            x = self.embedding(x)
            x, _ = self.encoder(x)
        x = self.classify(x)
        if y is not None:
            loss = self.loss(x.view(-1, x.shape[-1]), y.view(-1))
            return loss
        else:
            y_p = self.activation(x, dim=-1)
            return torch.argmax(y_p, dim=-1)


"""
优化器选择
"""
def optim(config, model):
    if config["optim_type"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=config["lr"])
    else:
        return torch.optim.SGD(model.parameters(), lr=config["lr"])
