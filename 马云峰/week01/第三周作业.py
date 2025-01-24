#coding:utf-8

import torch
import torch.nn as nn
import numpy as np
import random
import json
from matplotlib import pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(input_size=vector_dim, hidden_size=vector_dim, batch_first=True)  # RNN层
        self.classify = nn.Linear(vector_dim, sentence_length)  # 线性层
        self.loss_function = nn.CrossEntropyLoss()  # 交叉熵损失函数

    def forward(self, x, y=None):
        x = self.embedding(x)                      # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x, _ = self.rnn(x)                         # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, vector_dim)
        x = x[:, -1, :]                            # 取最后一个时间步的输出
        x = self.classify(x)                       # (batch_size, vector_dim) -> (batch_size, sentence_length)
        if y is not None:
            return self.loss_function(x, y)        # 预测值和真实值计算损失
        else:
            return x                               # 输出预测结果

# 建立字表
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 随便挑选一些字
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1           # 每个字对应一个序号
    vocab['unk'] = len(vocab)             # 未知字符用0表示
    return vocab

# 随机生成样本
# def build_sample(vocab, sentence_length):
#     x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
#     y = random.randint(0, sentence_length - 1)  # 随机选择一个位置作为目标位置
#     x = [vocab.get(word, vocab['unk']) for word in x]   # 将字转换成序号，为了做embedding
#     return x, y
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    #指定哪些字出现时为正样本
    if set("abc") & set(x):
        y = 1
    #指定字都未出现，则为负样本
    else:
        y = 0
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, y

# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

# 测试代码
def evaluate(model, vocab, sample_length):
    x, y = build_dataset(200, vocab, sample_length)   # 建立200个用于测试的样本
    print("本次预测集中共有%d个样本" % (len(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 假设y_pred是模型的输出
            y_p = torch.argmax(y_p)  # 获取最大概率的索引
            if int(y_p) == int(y_t):
                correct += 1  # 正确预测个数
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    epoch_num = 10        # 训练轮数
    batch_size = 20       # 每次训练样本个数
    train_sample = 500    # 每轮训练总共训练的样本总数
    char_dim = 20         # 每个字的维度
    sentence_length = 6   # 样本文本长度
    learning_rate = 0.005 # 学习率
    vocab = build_vocab()
    model = build_model(vocab, char_dim, sentence_length)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) # 构造一组训练样本
            optim.zero_grad()    # 梯度归零
            loss = model(x, y)   # 计算loss
            loss.backward()      # 计算梯度
            optim.step()         # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), "model.pth")
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)     # 建立模型
    model.load_state_dict(torch.load(model_path))             # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()   # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测位置：%d" % (input_string, int(result[i]))) # 打印结果

if __name__ == "__main__":
    main()
    test_strings = ["abcdef", "bcdefg", "cdefgh", "defghi"]
    predict("model.pth", "vocab.json", test_strings)
