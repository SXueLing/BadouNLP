import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt


"""

基于pytorch的网络编写
做一个多分类任务，判断特定字符在字符串的第几个位置，使用rnn和交叉熵。
"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab_size, num_classes):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vector_dim, padding_idx=0)
        self.rnn = nn.LSTM(input_size=vector_dim, hidden_size=vector_dim, batch_first=True, bidirectional=True)
        self.classify = nn.Linear(vector_dim * 2, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, seq_len) -> (batch_size, seq_len, vector_dim)
        x, _ = self.rnn(x)  # (batch_size, seq_len, vector_dim) -> (batch_size, seq_len, vector_dim*2)
        logits = self.classify(x)  # (batch_size, seq_len, vector_dim*2) -> (batch_size, seq_len, num_classes)
        if y is not None:
            loss = self.loss(logits.view(-1, logits.size(-1)), y.view(-1))  # Flatten for cross-entropy
            return loss
        else:
            return logits.argmax(dim=-1)  # Return predicted classes


def build_sample(vocab, sentence_length, target_chars):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    y = [vocab[char] if char in target_chars else 0 for char in x]  # Encode target characters
    x = [vocab.get(char, vocab['unk']) for char in x]  # Encode all characters
    return x, y


def build_dataset(sample_length, vocab, sentence_length, target_chars):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, sentence_length, target_chars)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_model(vocab, char_dim, sentence_length, num_classes):
    model = TorchModel(char_dim, sentence_length, len(vocab), num_classes)
    return model


def evaluate(model, vocab, sample_length, target_chars, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length, target_chars)
    correct, total = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        correct = (y_pred == y).sum().item()
        total = y.numel()
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab


def main():
    epoch_num = 10
    batch_size = 20
    train_sample = 500
    char_dim = 20
    sentence_length = 6
    learning_rate = 0.005
    target_chars = "你我他"  # Target characters for classification
    vocab = build_vocab()
    model = build_model(vocab, char_dim, sentence_length, len(vocab))
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        for _ in range(train_sample // batch_size):
            x, y = build_dataset(batch_size, vocab, sentence_length, target_chars)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / (train_sample // batch_size):.4f}")
        evaluate(model, vocab, sentence_length, target_chars, sentence_length)

if __name__ == "__main__":
    main()
