#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
用做一个多分类任务，判断特定字符在字符串的第几个位置，使用rnn和交叉熵。
（判断‘你’出现的索引位置，如果没出现那么返回-1）
"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding层
        self.layer = nn.RNN(vector_dim, 64, bias=False, batch_first=True)  # bias是说是否需要参数b
        self.classify = nn.Linear(64, sentence_length)  # 线性层
        self.loss = nn.functional.cross_entropy  #loss函数采用交叉熵

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim) 离散值转换为向量 维度是vector_dim 20 25 10
        x,h = self.layer(x)        #20 25 64                  #(batch_size, vector_dim) -> nn.Linear(vector_dim, 1)  3*20 20*1 -> 3*1
        x = x[:, -1, :]            # 取最后一个时间步的 输出，调整维度 #20 64
        y_pred = self.classify(x)  # (batch_size, vector_dim, 1) -> (batch_size, vector_dim) 20 25
        if y is not None:
            mask = (y != -1)       # 创建掩码，标记出有效位置索引的标签
            valid_y = y[mask]      # 选取有效位置索引的标签
            valid_y_pred = y_pred[mask]  # 选取对应预测值
            return self.loss(valid_y_pred, valid_y)
        else:
            return y_pred          #输出预测结果

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    y = -1  # 初始化为最大位置索引，表示没出现“你”字
    for index_temp, element in enumerate(x):
        if element == "你":
            y = index_temp
            break
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    sumY = 0
    for element in y:
        if element == -1:
            sumY += 1
    print("本次预测集中共有%d个正样本，%d个负样本"%(200-sumY, sumY))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_t, y_p in zip(y, y_pred):
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/200))
    return correct/(correct+wrong)


def main():
    #配置参数
    epoch_num = 20        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 2000    #每轮训练总共训练的样本总数
    char_dim = 10         #每个字的维度
    sentence_length = 25   #样本文本长度
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
            x, y = build_dataset(batch_size, vocab, sentence_length) #构造一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 10  # 每个字的维度
    sentence_length = 25  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    print("模型预测结果", result)  # 打印结果



if __name__ == "__main__":
    main()
    #test_strings = ["fn你feefnvfeefnvfeeqqfeeqqf", "你wzdfgfnvfeefnvfeeqqfeeqqf", "rqwdegfnvfeefnvfeeqqfeeqqf", "n我kwwwfnvfeefnvfeeqqfeeqqf"]
    #predict("model.pth", "vocab.json", test_strings)
