import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

'''
输入一个句子，判断特定字符出现在第几个位置，则该字符就属于哪一类。
句子长度：8
每个字扩展的向量维数：6
词表中字的个数：自己指定
'''
class NLPDivisionModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(NLPDivisionModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.pool = nn.AvgPool1d(sentence_length)  # 池化层
        self.classifier = nn.RNN(input_size=vector_dim, hidden_size=9, bias=True, batch_first=True)
        self.loss = nn.CrossEntropyLoss()  # 用交叉熵计算损失

    def forward(self, x, y=None):
        x = self.embedding(x)
        y_detail, y_pred = self.classifier(x)
        y_pred = torch.squeeze(y_pred)

        # x = x.transpose(1, 2)  # 交换第二个和第三个维度,使得池化针对句子长度
        # y_pred = self.pool(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


# 构建词表的函数
def build_vocab():
    chars = "你bcdefghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab

# 构建样本的函数
def build_sample(vocab, sentence_length):
    # 选用sample函数避免字重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]

    for index in range(len(x)):
        y = 8  # 未出现‘你’字符的单独作为一类
        if x[index] == '你':
            y = index
            break

    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字符转换成序号，方便后续的embedding操作  good
    return x, y

# 构建数据集的函数
def build_dataset(num_of_samples, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for num in range(num_of_samples):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)

    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 建立模型的函数
def build_model(vocab, vector_dim, sentence_length):
    model = NLPDivisionModel(vector_dim, sentence_length, vocab)
    return model

# 测试模型的函数
def evaluate_model(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(100, vocab, sentence_length)
    print("本次预测集中公有%d个样本" % 100)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model.forward(x)
        for y_p, y_t in zip(y_pred, y):
            _, p = torch.max(y_p, dim=0)
            if p == y_t:
                correct += 1
            else:
                wrong += 1
        acc = correct / 100
        print("正确个数目：%d,正确率：%.2f" % (correct, acc))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    num_of_samples = 500  # 每轮训练总共训练的样本总数
    char_dim = 6  # 每个字的维度
    sentence_length = 8  # 样本文本长度
    learning_rate = 0.005  # 学习率

    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(num_of_samples // batch_size):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate_model(model, vocab, sentence_length)  # 测试本轮模型结果
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


# 模型预测函数
def predict(model_path, vocab_path, input_strings):
    char_dim = 6
    sentence_length = 8
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
        print(result)
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d" % (input_string, torch.argmax(result)))  # 打印结果

if __name__ == '__main__':
    main()
    test_strings = ["fnvfeec你", "wz你dfghi", "rqwd你egd", "oonvkwww"]
    predict("model.pth", "vocab.json", test_strings)
