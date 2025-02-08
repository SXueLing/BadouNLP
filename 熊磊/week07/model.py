import torch
from torch import nn
from torch.optim import Adam, SGD
from transformers import BertModel, BertTokenizer
from config import Config

class clsposModel(nn.Module):
    def __init__(self, Config):
        super().__init__()
        vocab_size = Config['vocab_size']
        embedding_dim = Config['embedding_dim']
        model_type = Config['model_type']
        hidden_size = Config['hidden_size']
        num_layers = Config['num_layers']
        self.pooling_style = Config['pooling_style']
        self.use_bert = False
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        if model_type == 'bert':
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(Config['bert_path'], return_dict=False)
        elif model_type == 'fast_text':
            self.encoder = lambda x:x
        elif model_type == 'lstm':
            self.encoder = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        elif model_type == 'gru':
            self.encoder = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
        elif model_type == 'rnn':
            self.encoder = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        elif model_type == 'cnn':
            self.encoder = CNN(Config)

        self.clasify = nn.Linear(hidden_size, 2)
        self.loss = nn.functional.cross_entropy
    
    def forward(self, x, y = None):
        if self.use_bert == True:
            x = self.encoder(x)
        else:
            x = self.embedding(x)
            x = self.encoder(x)
        
        if isinstance(x, tuple):
            x = x[0]

        if self.pooling_style == 'max':
            self.pooling = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling = nn.AvgPool1d(x.shape[1])
        x = self.pooling(x.transpose(1, 2)).squeeze()

        y_pre = self.clasify(x)

        if y is not None:
            return self.loss(y_pre, y.squeeze())
        else:
            return y_pre



class CNN(nn.Module):
    def __init__(self, Config):
        super().__init__()
        hidden_size = Config['hidden_size']
        kernel_size = Config['kernel_size']
        pad = int((kernel_size - 1) / 2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)
    
    def forward(self, x):
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)
    
def choose_optimizer(Config, model):
    optimizer = Config['optimizer']
    if optimizer == 'adam':
        return Adam(model.parameters(), lr = Config['learning_rate'])
    elif optimizer == 'sgd':
        return SGD(model.parameters(), lr = Config['learning_rate'])
    return None
