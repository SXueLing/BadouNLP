import json

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
判断文本是否包含你我他，包含是正文本，否则是负文本
"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, str_length, vocab):
        super(TorchModel, self).__init__()
        self.embeding = nn.Embedding(len(vocab), vector_dim, padding_idx=0) # embeding层
        self.pool = nn.AvgPool1d(str_length)  # 池化层
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)  #rnn
        self.linear = nn.Linear(vector_dim, str_length)  # 线性层
        self.activation = torch.sigmoid # 激活函数
        self.loss = nn.functional.cross_entropy  # 损失函数

    def forward(self, x, y=None):
        x = self.embeding(x)            # 4*6*5
        x = x.transpose(1, 2)           #  4*5*6
        x = self.pool(x)
        x = x.squeeze()                  # 4*5

        rnn_out, hidden = self.rnn(x)
        x = rnn_out[:, -1, :]
        x = self.linear(x)
        y_pre = self.activation(x)
        if y is not None:
            return self.loss(y_pre, y)
        else:
            return y_pre


def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab["unk"] = len(vocab)
    return vocab

def build_sample(vocab, str_lenth):
    x = np.random.choice(list(vocab.keys()), str_lenth)
    if "a" in x:
        y = x.index("a")
    else:
        y = str_lenth
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

def build_dataset(vocab, sample_length ,str_lenth):
    X = []
    Y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, str_lenth)
        X.append(x)
        Y.append([y])
    return torch.LongTensor(X), torch.FloatTensor(Y)

def tain():
    str_lenth = 6
    vector_dim = 5
    eporch_num = 20
    batch_size = 50
    total_size = 500

    lr = 0.01
    vocab = build_vocab()
    model = TorchModel(vector_dim, str_lenth, vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log = []
    for epoch in range(eporch_num):
        model.train()
        watch_loss = []
        for i in range(total_size//batch_size):
            train_x, train_y = build_dataset(vocab, batch_size, str_lenth)
            optimizer.zero_grad()
            loss = model.forward(train_x, train_y)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        print("=========第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = test(vocab, str_lenth, model)
        log.append([acc, np.mean(watch_loss)])
    torch.save(model.state_dict(), 'model.pt')
    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()

    writer = open("vocab.txt", "w", encoding="utf-8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

def test(vocab, str_length ,model):
    model.eval()
    x, y = build_dataset(vocab, 200, str_length)
    print("本次预测集中共有%d个样本"%(len(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

if __name__ == '__main__':
    tain()
