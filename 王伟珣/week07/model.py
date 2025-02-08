import torch
import torch.nn as nn
from transformers import BertModel


class TextClassifyModel(nn.Module):
    def __init__(self, config):
        super(TextClassifyModel, self).__init__()
        model_type = config['model_type']
        n_classes = config['n_classes']
        vocab_size = config['vocab_size']
        hidden_size = config['hidden_size']
        num_layers = config['num_layers']

        self.use_embedding = True
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        
        if model_type == "bert":
            self.use_embedding = False
            self.encoder = BertModel.from_pretrained(config['pretrain_bert_model_path'], return_dict=False, ignore_mismatched_sizes=True)
            hidden_size = self.encoder.config.hidden_size
        elif model_type == 'lstm':
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == 'gru':
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == 'rnn':
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        
        self.pool_type = config['pool_type']
        self.linear = nn.Linear(hidden_size, n_classes)
        self.loss = nn.functional.cross_entropy
        return
    

    def forward(self, x, y=None):
        if not self.use_embedding:
            x = self.encoder(x)
        else:
            x = self.embedding(x)
            x = self.encoder(x)
        
        if isinstance(x, tuple):
            x = x[0]

        if self.pool_type == 'max':
            self.pool_layer = nn.MaxPool1d(x.shape[1])
        else:
            self.pool_layer = nn.AvgPool1d(x.shape[1])
        
        x = self.pool_layer(x.transpose(1, 2)).squeeze()

        pred = self.linear(x)

        if y is not None:
            return self.loss(pred, y.squeeze())
        else:
            return pred


def choose_optimizer(config, model):
    optimizer = config['optimizer']
    lr = config['learning_rate']
    if optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr)
    return None
