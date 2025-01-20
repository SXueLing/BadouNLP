# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
"""
Establishing the network model structure
"""

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    # Input is the problem character encoding
    def forward(self, x):
        x = self.embedding(x)
        # use lstm
        # x, _ = self.lstm(x)
        # use linear layer
        x = self.layer(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        # self.loss = nn.CosineEmbeddingLoss() #  Replace CosineEmbeddingLoss with Triplet Loss, because CosineEmbeddingLoss is designed to measure the similarity between two vectors (e.g., sentence embeddings) using cosine similarity.

    # Calculate the cosine distance 1-cos(a,b)
    # When cos=1, the two vectors are the same and the cosine distance is 0; when cos=0, the two vectors are orthogonal and the cosine distance is 1
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    def cosine_triplet_loss(self, anchor, positive, negative, margin=0.1):
        """
       Forward pass for the Siamese Network.
       If positive and negative are provided, compute the triplet loss.
       Otherwise, return the embedding of the anchor.
       """
        distance_positive = self.cosine_distance(anchor, positive)
        distance_negative = self.cosine_distance(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + margin)
        return torch.mean(losses)

    #sentence : (batch_size, max_length)
    def forward(self, anchor, positive=None, negative=None):
        """
        Forward pass for the Siamese Network.
        If positive and negative are provided, compute the triplet loss.
        Otherwise, return the embedding of the anchor.
        """
        anchor_embedding = self.sentence_encoder(anchor)
        if positive is not None and negative is not None:
            positive_embedding = self.sentence_encoder(positive) #vec:(batch_size, hidden_size)
            negative_embedding = self.sentence_encoder(negative)
            return self.cosine_triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
        else:
            return anchor_embedding


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config

    # Update config for testing
    Config["vocab_size"] = 10
    Config["max_length"] = 4

    # Initialize model
    model = SiameseNetwork(Config)

    # Dummy input data (anchor, positive, negative)
    anchor = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    positive = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    negative = torch.LongTensor([[4, 3, 2, 1], [0, 1, 2, 3]])

    # Forward pass with triplet loss
    loss = model(anchor, positive, negative)

    print("Triplet Loss:", loss.item())