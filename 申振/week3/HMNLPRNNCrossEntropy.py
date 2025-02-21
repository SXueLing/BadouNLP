import json
import random

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

"""
vector_dim:向量维度
sentence_length: 句子长度
vocab:词表
模型流程：
1.embedding
2.池化层
3.线性层
4.归一化
"""
class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length,vocab):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim,padding_idx=0)
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True,bias=False)
        self.layer = nn.Linear(vector_dim, sentence_length+1)
        self.loss = nn.CrossEntropyLoss()
    def forward(self, x,y=None):
        x = self.embedding(x)
        print(x.size())
        rnn_out, _ = self.rnn(x)
        x = rnn_out[:, -1, :]
        print(x.size())
        y_pred = self.layer(x)
        print(y_pred.size())
        if y is not None:
            return self.loss(y_pred, y)
        return y_pred

def build_sample(vocab,sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]

    try:
        y = x.index("你")
    except ValueError:
        y = sentence_length
    x = [vocab.get(word,vocab['unk']) for word in x]
    return x,y

def build_dataset(sample_length,sequence_length,vocab):
    X,Y = [],[]
    for i in range(sample_length):
        x,y = build_sample(vocab,sequence_length)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X),torch.LongTensor(Y)

def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab

def evaluate(model,vocab,sentence_length):
    model.eval()
    test_sample_num = 10
    x,y = build_dataset(test_sample_num,sentence_length,vocab)
    correct,wrong = 0,0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, x_t,y_t in zip(y_pred, x,y):  #与真实标签进行对比
            print(y_p,x_t, y_t)
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def predict(model_path, vocab_path, test_strings):
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = TorchModel(char_dim,sentence_length,vocab)
    model.load_state_dict(torch.load(model_path))
    x = []
    print(vocab)
    for test_string in test_strings:
        x.append([vocab[char] for char in test_string])
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, test_string in enumerate(test_strings):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (test_string, torch.argmax(result[i]), result[i])) #打印结果


def main():
    # 配置参数
    epoch_num = 10
    batch_size = 20
    learning_rate = 1e-3
    train_sample = 1000
    char_dim = 20
    sentence_length = 6
    # 建立词表
    vocab = build_vocab()
    model = TorchModel(char_dim, sentence_length, vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, sentence_length, vocab)
            model.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
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
if __name__ == "__main__":
    main()
    test_strings = ["你ndshj", "qwhe你j", "pshjdk", "qwezjm"]
    predict("model.pth", "vocab.json", test_strings)

