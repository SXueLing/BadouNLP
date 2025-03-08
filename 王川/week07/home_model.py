import torch
import torch.nn as nn
from transformers import BertModel


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        class_num = config["class_nums"]
        self.embedding = nn.Embedding(vocab_size + 1, hidden_size, padding_idx=0)
        self.use_bert = False
        if model_type == "bert":
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        elif model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "gated_cnn":
            self.encoder = GatedCNN(config)

        self.linear = nn.Linear(hidden_size, class_num)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        if self.use_bert:
            x = self.encoder(x)
        else:
            x = self.embedding(x)
            x = self.encoder(x)
        if isinstance(x, tuple):
            x = x[0]
        if self.pooling_style == "max":
            pooling_layer = nn.MaxPool1d(x.shape[1])
        else:
            pooling_layer = nn.AvgPool1d(x.shape[1])

        x = pooling_layer(x.transpose(1, 2)).squeeze()
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y.squeeze())
        else:
            return y_pred


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        kernel_size = config["kernel_size"]
        hidden_size = config["hidden_size"]
        pad = (kernel_size - 1) // 2
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=pad)

    def forward(self, x):
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)


class GatedCNN(nn.Module):
    def __init__(self, config):
        super(GatedCNN, self).__init__()
        self.cnn = CNN(config)
        self.gate = CNN(config)

    def forward(self, x):
        a = self.cnn(x)
        b = self.gate(x)
        b = torch.sigmoid(b)
        return torch.mul(a, b)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    else:
        return torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
