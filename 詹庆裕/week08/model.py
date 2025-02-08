import torch
import torch.nn as nn
from config import Config

class modelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vocab_size = Config["vocab_size"]
        hidden_size = Config["hidden_size"]
        self.embedding = nn.Embedding(vocab_size,hidden_size,padding_idx=0)
        if Config["model_type"] in ["lstm", "rnn", "gru"]:
            self.rnn_class = {
                "lstm": nn.LSTM,
                "rnn": nn.RNN,
                "gru": nn.GRU
            }[Config["model_type"]]
            num_layers = Config["num_layers"]
            self.bidirectional = Config["bidirectional"]
            self.rnn = self.rnn_class(
                hidden_size, hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional = self.bidirectional)
            self.classify = nn.Linear(hidden_size * 2, hidden_size)
        self.linear = nn.Sequential(
                      nn.Dropout(Config.get("dropout", 0.1)),
                      nn.Linear(hidden_size, hidden_size))
        self.pooling = {
            "max": nn.AdaptiveMaxPool1d(1),
            "avg": nn.AdaptiveAvgPool1d(1)
        }[Config["pooling_type"]]

    def forward(self, x):
        x = self.embedding(x)
        if hasattr(self, "rnn_class"):
            x, _ = self.rnn(x)
            if self.bidirectional:
                x = self.classify(x)
        else:
            x = self.linear(x)
        x = x.transpose(1, 2)
        x = self.pooling(x).squeeze(-1)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = modelEncoder()
        self.loss = {
            "cosEmbedding_loss": nn.CosineEmbeddingLoss(),
            "triplet_loss": nn.TripletMarginLoss()
        }[Config["loss_type"]]


    def forward(self, x1, x2=None, label=None):
        x1 = self.encoder(x1)
        if x2 is not None:
            x2 = self.encoder(x2)
            if Config["loss_type"] == "triplet_loss":
                label = self.encoder(label)
            return self.loss(x1, x2, label)
        return x1


def choose_optimizer(model):
    return {
        "adam":torch.optim.Adam(model.parameters(), lr=Config["lr"]),
        "sgd":torch.optim.SGD(model.parameters(), lr=Config["lr"])
        }[Config["optim_type"]]
