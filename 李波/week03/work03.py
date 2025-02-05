
# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
作业描述：nlp多分类，寻找特定字符位置

"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        # 第一个参数是字符表长度，第二个参数是向量维度
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        # 池化层 1d表示池化一维
        self.pool = nn.AvgPool1d(sentence_length)  # 池化层
        self.rnn = nn.RNN(vector_dim, vector_dim, bias=False, batch_first=True)

        self.classify = nn.Linear(vector_dim, sentence_length + 1)
        self.loss = nn.CrossEntropyLoss()  # 交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        # 由于池化是对最后一维进行操作，这里需要将sen_len和vector_dim转置
        # x = x.transpose(1, 2)  # (batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        # # 池化
        # x = self.pool(x)  # (batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        # x = x.squeeze()  # (batch_size, vector_dim, 1) -> (batch_size, vector_dim)
        rnn_out, hidden = self.rnn(x)  # (batch_size, vector_dim) -> (batch_size, 1) 3*5 5*1 -> 3*1
        x = rnn_out[:, -1, :]  # 或者写hidden.squeeze()也是可以的，因为rnn的hidden就是最后一个位置的输出
        # 接线性层做分类
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 字符集随便挑了一些字，实际上还可以扩充
# 为每个字生成一个标号
# {"a":1, "b":2, "c":3...}
# abc -> [1,2,3]
def build_vocab():
    chars = "abcdefghijk"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab


# 随机生成一个样本
# 从所有字中选取sentence_length个字
# 反之为负样本
def build_sample(vocab, sentence_length):
    # 注意这里用sample，是不放回的采样，每个字母不会重复出现，但是要求字符串长度要小于词表长度
    x = random.sample(list(vocab.keys()), sentence_length)
    # 指定哪些字出现时为正样本
    if "a" in x:
        y = x.index("a")
    else:
        y = sentence_length
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y


# 建立数据集
# 输入需要的样本数量。需要多少生成多少
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
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本
    print("本次预测集中共有%d个样本"%(len(y)))
    correct, wrong = 0, 0

    with torch.no_grad():  # 告诉模型不计算梯度，降低模型计算时间-> with用法
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比 y_pred是一个5维向量-[1,2,3,4,5] y是一个索引/类别 0 或 1 或 2 类
            # if np.argmax(y_p) == y_t:
            if torch.argmax(y_p) == int(y_t):
                # print('evaluate.right: y_p=%s\ty_t=%s' % (y_p,y_t))
                correct += 1
            else:
                # print('evaluate.wrong: y_p=%s\ty_t=%s' % (y_p, y_t))
                wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    #配置参数
    epoch_num = 20        #训练轮数
    batch_size = 40       #每次训练样本个数
    train_sample = 1000    #每轮训练总共训练的样本总数
    char_dim = 30         #每个字的维度
    sentence_length = 10   #样本文本长度
    learning_rate = 0.001 #学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
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

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 30  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string, torch.argmax(result[i]), result[i])) #打印结果


if __name__ == "__main__":
    main()
    test_strings = ["kijabcdefh", "gijkbcdeaf", "gkijadfbec", "kijhdefacb","kijhdefbcb"]
    predict("model.pth", "vocab.json", test_strings)
