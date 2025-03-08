"""
使用交叉熵 预测分类
指定的数据在第几个就是第几类
"""
import json
import random

import numpy as np
import torch
from torch import nn


class RnnModel(nn.Module):
    def __init__(self, vector_dim, hidden_size, output_size):
        super(RnnModel, self).__init__()
        self.embedding = nn.Embedding(vector_dim, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.embedding(x)
        x_out, h_out = self.rnn(x)
        x = x_out[:, -1, :]
        pred_y = self.linear(x)
        if y:
            return self.loss(pred_y, y)
        return pred_y


vocabulary = dict()
chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789"
key_word = "A"


def build_vocabulary():
    vocabulary["<pad>"] = 0
    for index, char in enumerate(chars):
        vocabulary[char] = index + 1
    vocabulary["<unk>"] = len(vocabulary) + 1


def build_sentence(sentence_length: int):
    rand_sentence = random.sample(chars, sentence_length)
    x = [vocabulary.get(x, vocabulary.get("<unk>")) for x in rand_sentence]
    if key_word in rand_sentence:
        return torch.LongTensor(x), torch.LongTensor(rand_sentence.index(key_word) + 1)
    return torch.LongTensor(x), torch.LongTensor(0)


def build_batch_sentence(sentence_length: int, batch_size: int):
    X, Y = [], []
    for _ in range(batch_size):
        x, y = build_sentence(sentence_length)
        X.append(x)
        Y.append(y)
    return X, Y


if __name__ == "__main__":
    build_vocabulary()
    sentence_length = 32
    batch_size = 6
    epoch = 300
    model = RnnModel(vector_dim=64, hidden_size=64, output_size=len(chars))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epoch):
        X, Y = build_batch_sentence(sentence_length, batch_size)
        all_loss = []
        for x, y in zip(X, Y):
            loss = model(x, y)
            optimizer.step()
            loss.backward()
            optimizer.zero_grad()
            all_loss.append(loss.item())
        all_loss = np.mean(all_loss)
    
    with open(file="./vocabulary.json", mode="w", encoding="utf-8") as f:
        f.write(json.dumps(vocabulary))
    
    torch.save(model.state_dict(), "./model.pt")
        
