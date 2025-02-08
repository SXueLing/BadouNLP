"""
NLP 多分类任务，
判断特定字符在字符串的第几个位置
"""
import json
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


class TorchModel(nn.Module):
    def __init__(self, char_dim, sentence_max_len, vocab, output_size=1):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), char_dim, padding_idx=0)
        self.pool = nn.AvgPool1d(sentence_max_len)  # 只会对最后一维做池化，所以需要先转置
        self.classify = nn.Linear(char_dim, sentence_max_len + 1)  # 全链接
        self.activation = torch.sigmoid
        self.loss = nn.functional.cross_entropy  # 交叉熵损失函数

    def forward(self, x, y=None):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze()
        x = self.classify(x)
        x = self.activation(x)
        if y is None:
            return x
        return self.loss(x, y)


def build_vocab(chars=None):
    chars = chars or "abcdefghijklmnopqrstuvwxyz"
    vocab = {'[pad]': 0}  # 长度不一样的时候填充此值到一样的长度
    for char in chars:
        vocab[char] = len(vocab)
    vocab['[unk]'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_max_length, target_char='e'):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_max_length)]
    if target_char in x:
        y = x.index(target_char)
    else:
        y = sentence_max_length
    # s = ''
    # for t in x:
    #     if t not in ['[pad]', '[unk]']:
    #         s += t
    # print(s, y)
    x = [vocab.get(word, vocab['[unk]']) for word in x]
    return x, y


def build_dataset(sample_length, vocab, sentence_max_length):
    dataset_x, dataset_y = [], []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_max_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_model(vocab, char_dim, sentence_max_length):
    model = TorchModel(char_dim, sentence_max_length, vocab)
    return model


def evaluate(model, vocab, sentence_max_length):
    model.eval()
    x, y = build_dataset(100, vocab, sentence_max_length)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p) == y_t:
                correct += 1
            else:
                wrong += 1
    correct_rat = correct / (correct + wrong)
    print(f"正确预测个数：{correct}, 正确率：{correct_rat}")
    return correct_rat


def train(model, epoch_num=100, batch_size=32, lr=0.001, train_sample_num=1000):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # parameters = list(model.parameters())
    sentence_max_length = 10  # len(parameters[0][0])
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample_num // batch_size):
            x, y = build_dataset(batch_size, vocab, sentence_max_length)
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_max_length)
        print("正确率：%f" % acc)
        log.append([acc, float(np.mean(watch_loss))])

    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), "nlp-multi-classification.pth")
    writer = open("nlp-multi-classification-vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


def predict(model_path, vocab_path, input_strings, char_dim, sentence_max_length, ):
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_max_length)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab.get('[unk]')) for char in input_string[:sentence_max_length]])
    for seq in x:
        if len(seq) < sentence_max_length:
            seq += [vocab['[pad]']] * (sentence_max_length - len(seq))
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        # print(f"输入：{input_string}, 预测类别：{round(float(result[i]))}, 概率值：{result[i]}")
        tag_index = torch.argmax(result[i])
        print(f"输入：{input_string}, 预测类别：{tag_index}, 概率值：{result[i][tag_index]}, {result[i]}")


if __name__ == '__main__':
    char_dim = 20
    sentence_max_length = 10
    lr = 0.005
    vocab = build_vocab()
    print(vocab)
    model = build_model(vocab, char_dim, sentence_max_length)

    train(model)

    # x, y = build_dataset(100, vocab, sentence_max_length=11)
    # print(x, y)

    strs = [
        'oxhiefpbpbd',
        'uvgmwqbfgdp',
        'ccwcigecl',
        'twsjjmnbn',
        'pmdggpzfsow',
        'akzjelxgrqu',
        'utbhslgsoj',
        'ugujyhclblv',
        'rwqwbdrhi',
    ]
    predict(
        "nlp-multi-classification.pth",
        "nlp-multi-classification-vocab.json",
        strs,
        char_dim,
        sentence_max_length
    )