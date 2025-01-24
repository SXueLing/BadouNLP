import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import json

"""
做一个多分类任务，判断特定字符在字符串的第几个位置，使用rnn和交叉熵。

"""

class TorchModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, text_length, hidden_size=128):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        # 全连接层
        self.linear = nn.Linear(hidden_size, text_length + 1)
        # 损失函数使用交叉熵-多分类问题
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)
        out, y_pred = self.rnn(x)
        y_pred = y_pred.squeeze(0)
        y_pred = self.linear(y_pred)
        if y is None:
            return y_pred
        else:
            return self.loss(y_pred, y)
        

def build_vocab():
    text = "你我他defghijklmnopqrstuvwxyz"
    vocab = {"pad" : 0}
    for i, char in enumerate(text):
        vocab[char] = i + 1
    vocab["unk"] = len(vocab)
    return vocab


def build_simple(vocab, text_length, target_char):
    x = [vocab.get(random.choice(list(vocab.keys())), vocab["unk"]) for _ in range(text_length)]
    target_idx = vocab.get(target_char, vocab["unk"])
    if target_idx in x:
        return x, x.index(target_idx)
    else:
        return x, text_length
    
def build_dataset(vocab, text_length, target_char, train_size):
    x, y = [], []
    for _ in range(train_size):
        x_, y_ = build_simple(vocab, text_length, target_char)
        x.append(x_)
        y.append(y_)
    return torch.LongTensor(x), torch.LongTensor(y)

# 评估模型
def evaluate(model, vocab, text_length, target_char):
    model.eval()
    x, y = build_dataset(vocab, text_length, target_char, 100)
    correct, wrong = 0, 0
    with torch.no_grad():
       res = model(x)
    for y_true, y_pred in zip(y, res):
        if y_true == y_pred.argmax():
            correct += 1
        else:
            wrong += 1
    print("正确率: {}".format(correct / (correct + wrong)))
    return correct / (correct + wrong)

def predict(text_length, input_strings):
    vector_dim = 10 # 输入特征维度
    vocab = json.load(open("vocab.json", "r", encoding="utf-8"))
    model = TorchModel(len(vocab), vector_dim, text_length)
    model.load_state_dict(torch.load("model.pt", weights_only=True))
    model.eval()
    x = []
    for input_string in input_strings:   
        x.append([vocab.get(word, vocab['unk']) for word in input_string])
    with torch.no_grad():
       res = model(torch.LongTensor(x))
    for x, y_pred in zip(input_strings, res):
        idx = y_pred.argmax()
        r = ""
        if idx == text_length: 
            r = "未找到"
        else:
            r = f"第{idx + 1}个"
        print("输入: {}, 模型预测: {}".format(x, r))
def main():
    epochs = 100 # 训练轮数
    batch_size = 20 # 每批训练样本数
    train_size = 500 # 训练样本数
    vector_dim = 10 # 输入特征维度
    text_length = 6 # 输入文本长度
    learning_rate = 0.001 # 学习率
    vocab = build_vocab()
    model = TorchModel(len(vocab), vector_dim, text_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    target_char = "你"
    # 训练数据
    for epoch in range(epochs):
        model.train()
        watch_loss = []
        for batch in range(train_size // batch_size):
            x, y = build_dataset(vocab, text_length, target_char, batch_size)
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward() # 反向传播
            optimizer.step()  # 更新参数
            watch_loss.append(loss.item())
        print("训练轮次: {}, loss: {}".format(epoch, np.mean(watch_loss)))
        acc = evaluate(model, vocab, text_length, target_char)
        log.append((epoch, np.mean(watch_loss), acc))   
    # 绘制损失函数曲线
    plt.plot(range(epochs), [i[1] for i in log], label="loss")
    plt.plot(range(epochs), [i[2] for i in log], label="acc")
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), "model.pt")
    writer = open("vocab.json", "w", encoding="utf-8")
    writer.write(json.dumps(vocab, ensure_ascii=False))

if __name__ == "__main__":
    main()
    test_strings = ["fnvfee", "你wzdfg", "rqdeg你", "n你kwww"]
    predict(6, test_strings)

