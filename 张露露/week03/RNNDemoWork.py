import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import json

"""
做一个多分类任务
判断特定字符在字符串的第几个位置
使用rnn和交叉熵
"""


class RNNWorkModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(RNNWorkModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)
        self.classify = nn.Linear(vector_dim, sentence_length + 1)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)
        rnn_out, hidden = self.rnn(x)
        x = rnn_out[:, -1, :]
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


# 生成字符集
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}
    for i, str in enumerate(chars):
        vocab[str] = i + 1
    vocab["unk"] = len(vocab)
    return vocab


# 创建一个样本
def build_sample(sentence_length, vocab):
    key_str = "k"
    x = random.sample(list(vocab.keys()), sentence_length)
    if key_str in x:
        y = x.index(key_str)
    else:
        y = sentence_length
    x = [vocab.get(word, vocab["unk"]) for word in x]
    return x, y


# 创建数据集
def build_dataset(total, sentence_length, vocab):
    X = []
    Y = []
    for i in range(total):
        x, y = build_sample(sentence_length, vocab)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X), torch.LongTensor(Y)


def evaluate(model, sentence_length, vocab):
    model.eval()
    x, y = build_dataset(200, sentence_length, vocab)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 错误预测个数：%d,正确率：%f" % (correct, wrong, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    epoch_num = 20
    sentence_length = 6
    batch_size = 20
    char_dim = 5
    total_num = 1000
    leaning_rate = 0.001
    vocab = build_vocab()
    log = []
    model = RNNWorkModel(char_dim, sentence_length, vocab)
    optim = torch.optim.Adam(model.parameters(), lr=leaning_rate)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for i in range(int(total_num / batch_size)):
            x, y = build_dataset(batch_size, sentence_length, vocab)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("======\n第%d轮平均loss：%s" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, sentence_length, vocab)
        log.append([acc, np.mean(watch_loss)])
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

    # torch.save(model.state_dict(),"RNNDemoWork.pth")

    # writer = open("vocab_work.json","w",encoding="utf8")
    # writer.write(json.dumps(vocab,ensure_ascii=False,indent=2))
    # writer.close()
    # 保存模型
    torch.save(model.state_dict(), "RNNDemoWork.pth")

    # 保存词表
    writer = open("vocab_work.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


def predict(model_path, vocab_json_path, input_strings):
    char_dim = 5
    sentence_length = 6
    vocab = json.load(open(vocab_json_path, "r", encoding="utf8"))
    model = RNNWorkModel(char_dim, sentence_length, vocab)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    x = []
    for string in input_strings:
        x.append([vocab[char] for char in string])
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
        for i,input_string in enumerate(input_strings):
            print("输入：%s,预测类别：%s,概率值：%s" % (input_string, torch.argmax(result[i]), result[i]))


if __name__ == "__main__":
    # main()
    input_strings = ["abgkhg", "kanwol", 'dajiah', "woaini", "seekan"]
    predict("RNNDemoWork.pth", "vocab_work.json", input_strings)
