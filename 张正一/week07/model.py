import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel
from config import Config
from loader import load_voacb

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config['hidden_size']
        num_layers = config['num_layers']
        class_num = config['class_num']
        vocab_size = config['vocab_size']
        model_type = config['model_type']
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.use_bert = False
        if model_type == 'bert':
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config['pretrain_model_path'], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        elif model_type == 'RNN':
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        elif model_type == 'LSTM':
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        elif model_type == 'GRU':
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.classify = nn.Linear(hidden_size, class_num)
        self.pooling_style = config['pooling_style']
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, x, target=None):
        if self.use_bert:
            x = self.encoder(x)
        else:
            x = self.embedding(x)
            x = self.encoder(x)
        if isinstance(x, tuple):
            x = x[0]
        if self.pooling_style == 'max':
            self.pooling_layer = nn.MaxPool1d(x.shape[1]) # (batch_size, hidden_size, seq_len)
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1]) # (batch_size, hidden_size, seq_len)
        x = x.transpose(1, 2) # (batch_size, seq_len, hidden_size) 
        x = self.pooling_layer(x) # (batch_size, hidden_size, 1)
        x = x.squeeze() # (batch_size, seq_len)
        predict = self.classify(x)
        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict
#优化器的选择
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
            

if __name__ == '__main__':
    Config['class_num'] = 2
    Config['vocab_size'] = len(load_voacb(Config['vocab_path']))
    torch_model = TorchModel(Config)