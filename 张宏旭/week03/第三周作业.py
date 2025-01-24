#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
多分类任务：
判断某个特定字符在字符串中的具体位置。
"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.LSTM(vector_dim, vector_dim, batch_first=True)  # LSTM层
        self.classify = nn.Linear(vector_dim, sentence_length)  # 全连接层，多分类
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x, _ = self.rnn(x)  # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, vector_dim)
        x = self.classify(x)  # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, sen_len)
        x = x.mean(dim=1)  # 聚合时间步的信息 -> (batch_size, sen_len)
        if y is not None:
            return self.loss(x, y)  # 计算损失
        else:
            return torch.softmax(x, dim=1)  # 输出预测结果


def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    target_char = random.choice("你我他")  # 指定目标字符
    if target_char in x:
        y = x.index(target_char)  # 标记目标字符的位置
    else:
        x[random.randint(0, sentence_length - 1)] = target_char
        y = x.index(target_char)
    x = [vocab.get(word, vocab['unk']) for word in x]  # 转换成序号
    return x, y


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


def evaluate(model, vocab, sample_length, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)  # 构造200个样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred.argmax(dim=1), y):
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率

    vocab = build_vocab()
    model = build_model(vocab, char_dim, sentence_length)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========")
        print("第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length, sentence_length)
        log.append([acc, np.mean(watch_loss)])

    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), "model_2.pth")
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()

def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)      # 建立模型
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab['unk']) for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        pred_class = result[i].argmax().item()  # 获取预测类别
        pred_prob = result[i].max().item()      # 获取最高概率值
        print(f"输入：{input_string}, 预测类别：{pred_class}, 概率值：{pred_prob:.6f}")  # 打印结果


if __name__ == "__main__":
    main()
    test_strings = ["他nvfee", "wz你dfg", "rqwde我", "n我kwww"]
    predict("model_2.pth", "vocab.json", test_strings)
