#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现，并判断其位置

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, num_classes):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.rnn = nn.RNN(vector_dim, 64, bias=False, batch_first=True)  # 修改为RNN层
        self.dropout = nn.Dropout(0.5)  # 设置丢弃概率为0.5，可根据实际调整
        # 线性层
        self.classify = nn.Linear(64, num_classes)
        # 交叉熵函数，用于计算损失
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # vector_dim 层
        x = self.embedding(x)
        _, h_n = self.rnn(x)
        x = h_n[-1]  # 取最后一个时间步的隐藏状态
        x = self.dropout(x)
        x = self.classify(x)
        if y is not None:
            return self.loss(x, y)
        else:
            return x

def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1
    vocab['unk'] = len(vocab)
    return vocab

def build_sample(vocab, sentence_length):
    specific_chars = ["你", "我", "他"]
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    positions = [i+1 for i, char in enumerate(x) if char in specific_chars]
    if positions:
        y = positions[0]
    else:
        y = 0
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

def build_model(vocab, char_dim, sentence_length, num_classes):
    model = TorchModel(char_dim, sentence_length, vocab, num_classes)
    return model

def evaluate(model, vocab, sample_length, num_classes):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    y = y.squeeze()
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y != 0), sum(y == 0)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        _, predicted = torch.max(y_pred, 1)
        correct = (predicted == y).sum().item()
        wrong = y.size(0) - correct
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def predict(model_path, vocab_path, input_strings, sentence_length):
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    char_dim = 20
    num_classes = sentence_length + 1
    model = build_model(vocab, char_dim, sentence_length, num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载模型参数，并设置 weights_only=True
    x = []
    for input_string in input_strings:
        input_chars = list(input_string)
        if len(input_chars) > sentence_length:
            input_chars = input_chars[:sentence_length]
        elif len(input_chars) < sentence_length:
            input_chars = input_chars + ['pad'] * (sentence_length - len(input_chars))
        x_seq = [vocab.get(char, vocab['unk']) for char in input_chars]
        x.append(x_seq)
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
        _, predicted = torch.max(result, 1)
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d" % (input_string, predicted[i]))

def main():
    epoch_num = 10
    batch_size = 20
    train_sample = 200
    char_dim = 20
    sentence_length = 6  # 确保训练和预测时的 sentence_length 一致
    learning_rate = 0.005
    vocab = build_vocab()
    num_classes = sentence_length + 1
    model = build_model(vocab, char_dim, sentence_length, num_classes)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length, num_classes)
        log.append([acc, np.mean(watch_loss)])
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), "model.pth")
    writer = open("vocab词表.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()

if __name__ == "__main__":
    main()
    # test_strings = ["fnvfee", "你dfg", "rqwdeg", "我kwww"]
    # predict("model.pth", "vocab词表.json", test_strings, sentence_length=6)  # 确保训练和预测时的 sentence_length 一致
