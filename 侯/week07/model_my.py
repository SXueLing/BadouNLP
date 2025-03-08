"""
@Project ：cgNLPproject 
@File    ：model_my.py
@Date    ：2025/1/6 15:53 
"""
import torch

from config_my import Config
import torch.nn as nn
from transformers import BertModel

class MyTorchModule(nn.Module):
    def __init__(self, config):
        super(MyTorchModule, self).__init__()
        hidden_size = config['hidden_size']
        input_dim = config['input_dim']
        vocab_length = config['vocab_length'] + 1
        model_type = config['model_type']
        num_layers = config['num_layers']
        class_num = config['class_num']
        self.use_bert = False
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_length, embedding_dim=hidden_size, padding_idx=0)
        if model_type == 'fast_text':
            self.encoder = lambda x:x
        elif model_type == 'lstm':
            self.encoder = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        elif model_type == 'rnn':
            self.encoder = nn.RNN(input_size=hidden_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        elif model_type == 'gru':
            self.encoder = nn.GRU(input_size=hidden_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        elif model_type == 'cnn':
            self.encoder = CNN(config)
        elif model_type == 'gated_cnn':
            self.encoder = GATED_CNN(config)
        elif model_type == 'bert':
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config['pretrain_model_path'], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        elif model_type == 'bert_lstm':
            self.use_bert = True
            self.encoder = Bert_LSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == 'bert_cnn':
            self.use_bert = True
            self.encoder = Bert_CNN(config)
            hidden_size = self.encoder.bert.config.hidden_size

        self.linear = nn.Linear(in_features=hidden_size,out_features=class_num)
        self.pooling_style = config['pooling_style']
        self.loss = nn.CrossEntropyLoss()


    def forward(self, x_, y=None):
        if self.use_bert:
            x_ = self.encoder(x_)
        else:
            x_ = self.embedding_layer(x_)
            x_ = self.encoder(x_)

        if isinstance(x_, tuple):
            x_ = x_[0]

        if self.pooling_style == 'max':
            self.pooling_layer = nn.MaxPool1d(x_.shape[1])
        else:
            self.pooling_layer = nn.AvgPool1d(x_.shape[1])
        x_ = self.pooling_layer(x_.transpose(1, 2)).squeeze()
        # 也可以x_[:,-1,:]
        y_pred = self.linear(x_)
        if y is None:
            return y_pred
        else:
            return self.loss(y_pred, y.squeeze())

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        kernel_size = config['kernel_size']
        hidden_size = config['hidden_size']
        pad = int((kernel_size - 1)/2)
        self.cnn = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size,kernel_size=kernel_size,bias=False,padding=pad)

    def forward(self, x_):
        # x_ = batch_size, sentence_len, embedding_size ???
        return self.cnn(x_.transpose(1,2)).transpose(1,2)

class GATED_CNN(nn.Module):
    def __init__(self, config):
        super(GATED_CNN, self).__init__()
        self.cnn = CNN(config)
        self.gate = CNN(config)

    def forward(self,x_):
        cnn_x = self.cnn(x_)
        gate_x = self.gate(x)
        gate_x = torch.sigmoid(gate_x)
        return torch.mul(cnn_x, gate_x)

class Bert_LSTM(nn.Module):
    def __init__(self, config):
        super(Bert_LSTM, self).__init__()
        self.bert = BertModel.from_pretrained(config['pretrain_model_path'], return_dict=False)
        self.rnn = nn.LSTM(self.bert.config.hidden_size,self.bert.config.hidden_size,batch_first=True)

    def forward(self,x_):
        bert_x,_ = self.bert(x_)
        rnn_x, aa = self.rnn(bert_x)
        return rnn_x

class Bert_CNN(nn.Module):
    def __init__(self, config):
        super(Bert_CNN, self).__init__()
        self.bert = BertModel.from_pretrained(config['pretrain_model_path'], return_dict=False)
        config['hidden_size'] = self.bert.config.hidden_size
        self.cnn = CNN(config)

    def forward(self,x_):
        bert_x,_ = self.bert(x_)
        rnn_x = self.cnn(bert_x)
        return rnn_x

def choose_optimizer(config, model_):
    optimizer = config['optimizer']
    lr = config['lr']
    if optimizer == 'adam':
        return torch.optim.Adam(model_.parameters(), lr=lr)
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)


if __name__ == '__main__':
    model = MyTorchModule(Config)
    x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    label = torch.LongTensor([1, 2])
    sentence_out = model(x, label)
    print(x[1], type(x[1]), len(x[1]))
    print(sentence_out.shape)
    print(sentence_out)
    # print(pooler_output)
