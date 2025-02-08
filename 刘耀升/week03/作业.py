import torch
import torch.nn as nn
import random
import numpy as np
import json
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
"""
NLP分类任务改为多分类任务
找一个特定字符在哪，假如有6个字符a在第几个位置就是第几类
用RNN和交叉熵
"""
class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_len, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.RNN = nn.RNN(vector_dim, vector_dim, batch_first=True)
        self.linear = nn.Linear(vector_dim, sentence_len+1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)
        output, x = self.RNN(x)
        #print(f"RNNx:{x}  RNNShape:{x.shape}  RNNSqueeze:{x.squeeze(0)}")
        #RNN 返回x的shape为(1,20,20)需对第0层降维后传入线性层
        y_pre = self.linear(x.squeeze(0))
        if y is not None:
            return self.loss(y_pre, y)
        else:
            return y_pre

def build_vocab():
    vocab = {"[pab]": 0}
    string = "abcdefghijklmnopqrstuvwxyz"
    for i, char in enumerate(string):
        vocab[char] = i + 1
    vocab["[unk]"] = len(vocab)
    return vocab


def build_sample(vocab, sentence_len):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_len)]
    if "a" in x:
        y = x.index("a")
    else:
        y = sentence_len
    x = [vocab.get(char, vocab["[unk]"]) for char in x]
    return x, y


def build_dataest(sample_len, sentence_len, vocab):
    X = []
    Y = []
    for i in range(sample_len):
        x, y = build_sample(vocab, sentence_len)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X), torch.LongTensor(Y)


def model_test(model, vocab, sentence_len):
    model.eval()
    x, y = build_dataest(200, sentence_len, vocab)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pre = model(x)
        for y_p, y_t in zip(y_pre, y):
            if int(y_t) == y_p.argmax():
                correct += 1
            else:
                wrong += 1
    print(f"正确预测个数{correct},正确率{correct / (correct + wrong)}")
    return correct / (correct + wrong)


def main():
    train_num = 10  #训练轮数
    batch_size = 20  #每次训练样本个数
    train_sample = 500  #每轮训练总共样本个数
    char_dim = 20  #字符维度
    sentence_len = 6  #字符长度
    learning_rate = 0.005  #学习率
    vocab = build_vocab()
    model = TorchModel(char_dim, sentence_len, vocab)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    for epoch in range(train_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample // batch_size)):
            x, y = build_dataest(batch_size, sentence_len, vocab)
            model.zero_grad()
            loss = model.forward(x,y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss)}")
        acc = model_test(model, vocab, sentence_len)
        log.append([acc, np.mean(watch_loss)])
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_len = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  #加载字符表
    model = TorchModel(char_dim, sentence_len, vocab)
    model.load_state_dict(torch.load(model_path,weights_only=True))
    model.eval()
    x = []
    for input_string in input_strings:
        input_s = ([vocab[char] for char in input_string])
        if len(input_s) < sentence_len:
            input_s += [vocab["[pad]"] * sentence_len-len(input_s)]
        elif len(input_s) > sentence_len:
            input_s = input_s[:sentence_len]
        x.append(input_s)

    with torch.no_grad():
        result = model(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print(f"输入：{input_string}, 预测位置：第{result[i].argmax()+1 if result[i].argmax() < sentence_len else -1}位")  # 打印结果


if __name__ == "__main__":
    main()
    test_strings = ["favfee", "wzadfg", "aqwdeg", "nbkwww"]
    predict("model.pth", "vocab.json", test_strings)
