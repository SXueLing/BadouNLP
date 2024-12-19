# Correcting and saving the final properly formatted script

corrected_script = """
#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random

"""
# Adding a comment to describe the task
corrected_script += """
# 基于pytorch的网络编写
# 实现一个网络完成一个简单nlp任务
# 判断某个字符在字符串中的具体位置
"""

corrected_script += """
class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding layer
        self.rnn = nn.LSTM(input_size=vector_dim, hidden_size=128, batch_first=True)  # LSTM layer
        self.classify = nn.Linear(128, sentence_length)  # Linear layer for classification
        self.loss = nn.CrossEntropyLoss()  # Use cross-entropy loss for multi-class classification

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x, _ = self.rnn(x)     # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, 128)
        x = x[:, -1, :]        # Take the last time step output
        x = self.classify(x)   # (batch_size, 128) -> (batch_size, sentence_length)
        if y is not None:
            return self.loss(x, y)  # Compute loss
        else:
            return torch.argmax(x, dim=-1)  # Predict position index

def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  # Character set
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # Each character has a unique index
    vocab['unk'] = len(vocab)  # Unknown token
    return vocab

def build_sample(vocab, sentence_length=10):
    sentence = random.choices(list(vocab.keys()), k=sentence_length)
    target_char = random.choice(list(vocab.keys()))
    position = random.randint(0, sentence_length - 1)
    sentence[position] = target_char  # Place the target character
    x = [vocab.get(char, vocab['unk']) for char in sentence]  # Convert to indices
    y = position  # Label is the position index of the target character
    return x, y

def build_dataset(vocab, batch_size=32, sentence_length=10):
    dataset = [build_sample(vocab, sentence_length) for _ in range(batch_size)]
    x = torch.tensor([item[0] for item in dataset], dtype=torch.long)
    y = torch.tensor([item[1] for item in dataset], dtype=torch.long)
    return x, y

if __name__ == "__main__":
    epoch_num = 10  # Training epochs
    batch_size = 20  # Batch size
    train_sample = 1000  # Training samples
    vector_dim = 16  # Embedding dimensions
    sentence_length = 10  # Sentence length
    vocab = build_vocab()  # Vocabulary
    model = TorchModel(vector_dim, sentence_length, vocab)  # Instantiate model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimizer

    for epoch in range(epoch_num):
        model.train()
        train_loss = 0
        for _ in range(train_sample // batch_size):
            x, y = build_dataset(vocab, batch_size, sentence_length)
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epoch_num}, Loss: {train_loss:.4f}")

    model.eval()
    x, y = build_dataset(vocab, batch_size, sentence_length)
    predictions = model(x)
    print("Predictions:", predictions)
    print("Ground Truth:", y)
"""
