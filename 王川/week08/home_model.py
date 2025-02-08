import torch.nn as nn
import torch
from transformers import BertModel


class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        self.config = config
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx = 0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first = True)
        if config["model_type"] == "bert":
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict = False)
            hidden_size = self.encoder.config.hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        if self.config["model_type"] == "bert":
            x, _ = self.encoder(x)
        else:
            x = self.embedding(x)
            x, _ = self.lstm(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        x = self.linear(x)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.encoder = SentenceEncoder(config)

    def cosine_triplet_loss(self, a, p, n, margin=None):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin.squeeze()
        return torch.mean(diff[diff.gt(0)]) #greater than

    def cosine_distance(self, sen1, sen2):
        sen1 = nn.functional.normalize(sen1, dim = -1)
        sen2 = nn.functional.normalize(sen2, dim = -1)
        cosine = torch.sum(torch.mul(sen1, sen2), dim = -1)
        return 1 - cosine

    def forward(self, a, p = None, n = None):
        if p is not None and n is not None:
            a = self.encoder(a)
            p = self.encoder(p)
            n = self.encoder(n)
            return self.cosine_triplet_loss(a, p, n)
        else:
            return self.encoder(a)

def choose_optimizer(config, model):
    lr = config["learning_rate"]
    optim = config["optimizer"]
    if optim == "adam":
        return torch.optim.Adam(model.parameters(), lr= lr)
    else:
        return torch.optim.SGD(model.parameters(), lr= lr)
