import torch
from torch import nn
from torch.optim import Adam, SGD
from transformers import BertModel


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config['hidden_size']
        vocab_size = config['vocab_size']
        class_num = config['class_num']
        model_type = config['model_type']
        num_layers = config['num_layers']
        self.use_bert = False
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if model_type == 'fast_text':
            self.encoder = lambda x : x
        elif model_type == 'lstm':
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        elif model_type == 'gru':
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        elif model_type == 'rnn':
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        elif model_type == 'cnn':
            self.encoder = CNN(config)
        elif model_type == 'gated_cnn':
            self.encoder = GatedCNN(config)
        elif model_type == 'bert':
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        elif model_type == 'bert_lstm':
            self.use_bert = True
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size

        self.classifier = nn.Linear(hidden_size, class_num)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y = None):
        if self.use_bert:
            x = self.encoder(x)
        else:
            x = self.embedding(x)
            x = self.encoder(x)

        # 如果有多返回的数据 RNN类的
        if isinstance(x, tuple):
            x = x[0]

        if self.pooling_style == 'max':  #  batch_size, sen_len, input_dim
            self.pooling_type = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling_type = nn.AvgPool1d(x.shape[1])
        x = self.pooling_type(x.transpose(1, 2)).squeeze()
        pre = self.classifier(x)
        if y is not None:
            return self.loss(pre, y.squeeze())
        else:
            return pre

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1) / 2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)

    def forward(self, x):  # x : (batch_size, max_len, embeding_size)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)

class GatedCNN(nn.Module):
    def __init__(self, config):
        super(GatedCNN, self).__init__()
        self.cnn = CNN(config)
        self.grate = CNN(config)

    def forward(self, x):
        a = self.cnn(x)
        b = self.grate(x)
        b = torch.sigmoid(b)
        return torch.mul(a, b)

class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        x = self.bert(x)[0]
        x,_ = self.rnn(x)
        return x

#  优化器
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)

if __name__ == '__main__':
    print("model")
