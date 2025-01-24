# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config['hidden_size']
        vocab_size = config['vocab_size'] + 1
        
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x
    

class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
    
    def forward(self, a, p=None, n=None):
        if p is not None and n is not None:
            va = self.sentence_encoder(a)
            vp = self.sentence_encoder(p)
            vn = self.sentence_encoder(n)

            return self.cosine_triplet_loss(va, vp, vn)
        else:
            return self.sentence_encoder(a)
        
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine
    
    def cosine_triplet_loss(self, a, p, n, margin=None):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin.squeeze()
        return torch.mean(diff[diff.gt(0)])
    

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
