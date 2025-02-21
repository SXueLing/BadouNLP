import torch
import torch.nn as nn
from torch import optim
from transformers import BertModel


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.config = config
        vocab_size = config['vocab_size']
        input_dim = config['input_dim']
        bert_path = config['bert_path']
        # self.embedding = nn.Embedding(vocab_size, input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=config['num_layers'], batch_first=True)
        self.bert = BertModel.from_pretrained(bert_path, return_dict=False)
        self.classifier = nn.Linear(768, 21128)
        # self.dropout = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y = None):
        # x = self.embedding(x)
        # x, _ = self.layer(x)
        if y is not None:
            # mask attention
            mask = torch.tril(torch.ones(x.shape[0], x.shape[1], x.shape[1]))
            x, _ = self.bert(x, attention_mask=mask)
            pre = self.classifier(x)
            return self.loss(pre.view(-1, pre.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            pre = self.classifier(x)
            return torch.softmax(pre, dim=-1)


def choose_optimizer(config, model):
    learning_rate = config["learning_rate"]
    return optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == '__main__':
    from config import Config
    model = TorchModel(Config)
