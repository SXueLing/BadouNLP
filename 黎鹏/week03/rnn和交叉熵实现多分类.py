import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.LSTM(input_size=vector_dim, hidden_size=128, batch_first=True)  # LSTM层
        self.classify = nn.Linear(128, sentence_length + 1)  # 线性层，多分类（包含未出现的类别）
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x, _ = self.rnn(x)  # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, hidden_size)
        x = x[:, -1, :]  # 取最后一个时间步的输出
        x = self.classify(x)  # (batch_size, hidden_size) -> (batch_size, sentence_length+1)
        if y is not None:
            return self.loss(x, y)  # 返回损失值
        else:
            return torch.argmax(x, dim=1)  # 返回预测位置索引

def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab

def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    target_chars = "你"
    for idx, char in enumerate(x):
        if char in target_chars:
            y = idx + 1  # 目标字符的索引位置（从1开始）
            break
    else:
        y = 0  # 如果目标字符未出现，返回0
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

def evaluate(model, vocab, sample_length, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)
    correct = 0
    total = len(y)
    with torch.no_grad():
        y_pred = model(x)
        correct = (y_pred == y).sum().item()
    print(f"正确预测个数：{correct}, 正确率：{correct / total:.4f}")
    return correct / total

def main():
    epoch_num = 10
    batch_size = 20
    train_sample = 500
    char_dim = 20
    sentence_length = 6
    learning_rate = 0.005
    vocab = build_vocab()
    model = build_model(vocab, char_dim, sentence_length)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(train_sample // batch_size):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print(f"第{epoch + 1}轮平均loss: {np.mean(watch_loss):.4f}")
        acc = evaluate(model, vocab, train_sample, sentence_length)
        log.append([acc, np.mean(watch_loss)])
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), "model.pth")
    with open("vocab.json", "w", encoding="utf8") as writer:
        writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))

def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab['unk']) for char in input_string])
    model.eval()
    with torch.no_grad():
        result = model(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print(f"输入：{input_string}, 预测位置：{result[i].item()}")

if __name__ == "__main__":
    main()
    test_strings = ["fnv你他e", "wz你dfg", "rqwdeg", "n我kw你w"]
    predict("model.pth", "vocab.json", test_strings)
