# coding: utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, sentence_length, num_classes):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)  # num_classes包括所有可能的位置+1（不存在的情况）

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sentence_length) -> (batch_size, sentence_length, embedding_dim)
        _, (h_n, _) = self.rnn(x)  # 取出最后一个时间步的隐藏状态
        h_n = h_n[-1]  # 取最后一层的隐藏状态
        logits = self.fc(h_n)  # (batch_size, hidden_dim) -> (batch_size, num_classes)
        if y is not None:
            loss = nn.CrossEntropyLoss()(logits, y)  # 使用交叉熵损失
            return loss
        else:
            return logits  # 输出logits，后续进行softmax或其他处理

def build_vocab():
    chars = "你我他abcdefghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab

def build_sample(vocab, sentence_length, target_chars):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    positions = [i for i, char in enumerate(x) if char in target_chars]
    if positions:
        y = random.choice(positions)  # 随机选择一个目标字符的位置作为标签
    else:
        y = len(x)  # 如果不存在目标字符，则标记为一个特殊位置（例如字符串长度+1）
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

def build_dataset(sample_length, vocab, sentence_length, target_chars):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length, target_chars)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def evaluate(model, vocab, sample_length, target_chars):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length, target_chars)
    correct = 0
    total = 0
    with torch.no_grad():
        preds = model(x)
        preds = preds.argmax(dim=1)
        for pred, true in zip(preds, y):
            if true < len(x[0]):  # 如果真实标签不是特殊标记
                total += 1
                if pred == true:
                    correct += 1
    print(f"正确预测个数：{correct}, 正确率：{correct / total if total > 0 else 0}")
    return correct / total if total > 0 else 0

def main():
    epoch_num = 10
    batch_size = 20
    train_sample = 500
    embedding_dim = 20
    hidden_dim = 50
    sentence_length = 6
    learning_rate = 0.001
    target_chars = {'你', '我', '他'}

    vocab = build_vocab()
    num_classes = sentence_length + 1  # 包括所有可能的位置+1（不存在的情况）
    model = RNNModel(len(vocab), embedding_dim, hidden_dim, sentence_length, num_classes)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length, target_chars)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss)}")
        acc = evaluate(model, vocab, sentence_length, target_chars)
        log.append([acc, np.mean(watch_loss)])

    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), "rnn_model.pth")
    with open("vocab.json", "w", encoding="utf8") as writer:
        json.dump(vocab, writer, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
