import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import json

class torchModel(nn.Module):
    def __init__(self, vocab, vec_dim, hidden_size,output_size):
        super(torchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vec_dim, padding_idx = 0)
        self.rnn = nn.RNN(vec_dim, hidden_size, batch_first = True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y = None):
        x = self.embedding(x)
        output, h = self.rnn(x)
        h = h.squeeze(0)
        y_pred = self.linear(h)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"
    vocab = {"pad":0}
    for idx, char in enumerate(chars):
        vocab[char] = idx + 1
    vocab["unk"] = len(vocab)
    return vocab

def build_sample(vocab, s_len):
    x = [random.choice(list(vocab)) for _ in range(s_len)]
    if "你" in x:
        y = x.index("你")
    else:
        y = len(x)
    x = [vocab.get(s, vocab["unk"]) for s in x]
    return x, y

def build_dataset(total_nums, vocab, s_len):
    X = []
    Y = []
    for i in range(total_nums):
        x, y = build_sample(vocab, s_len)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X), torch.LongTensor(Y)

def evaluate(model, vocab, s_len):
    model.eval()
    test_nums = 50
    test_x, test_y = build_dataset(test_nums, vocab, s_len)
    correct = 0
    with torch.no_grad():
        y_pred = model(test_x)
    for y_p, y_t in zip(y_pred, test_y):
        label = y_p.argmax()
        if label == y_t:
            correct += 1
    print("测试总数为：%d，正确率为：%f" % (test_nums, correct / test_nums))
    return correct / test_nums

def main():
    epoch_nums = 15
    lr = 0.005
    batch_size = 10
    train_nums = 500
    vocab = build_vocab()
    s_len = 6
    vec_dim = 20
    hidden_size = 10
    output_size = s_len + 1
    log = []

    model = torchModel(vocab, vec_dim, hidden_size, output_size)
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    train_x, train_y = build_dataset(train_nums, vocab, s_len)

    for epoch in range(epoch_nums):
        model.train()
        watch_loss = []
        for idx in range(train_nums // batch_size):
            x = train_x[idx * batch_size : (idx + 1) * batch_size]
            y = train_y[idx * batch_size : (idx + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())

        print("第%d轮Epoch的loss：%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, s_len)
        log.append([acc, np.mean(watch_loss)])

    torch.save(model.state_dict(), "model.pth")
    with open("vocab.json", "w", encoding = "utf-8") as f:
        f.write(json.dumps(vocab, ensure_ascii = False, indent = 2))

    plt.plot(range(epoch_nums), [l[0] for l in log], label = "accuracy")
    plt.plot(range(epoch_nums), [l[1] for l in log], label = "loss")
    plt.legend()
    plt.show()

def predict(model_path, vocab_path, test_strings):
    vocab = json.load(open(vocab_path, "r", encoding = "utf-8"))
    hidden_size = 10
    s_len = 6
    output_size = s_len + 1
    vec_dim = 20
    model = torchModel(vocab, vec_dim, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    x = []
    for input_string in test_strings:
        tmp = [vocab.get(s, vocab["unk"]) for s in input_string][:s_len] # 多于sentence_len
        if len(tmp) < s_len:
            tmp += [vocab["pad"]] * (s_len - len(tmp)) # 少于sentence_len
        x.append(tmp)
    x = torch.LongTensor(x)
    with torch.no_grad():
        y_pred = model(x)
    for y_p, vec in zip(y_pred, test_strings):
        label = y_p.argmax().item()
        print("输入：%s，预测类别：%d" % (vec, label))


if __name__ == "__main__":
    main()
    test_strings = ["fnvfee", "wz你dfg", "r你wd", "n她kwwwp"] # 少于sentence_len和多于sentence_len均可预测
    predict("model.pth", "vocab.json", test_strings)
