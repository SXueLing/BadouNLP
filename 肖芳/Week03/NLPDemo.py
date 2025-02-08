#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from build_sample import *

"""

尝试修改nlpdemo，做一个多分类任务，判断特定字符在字符串的第几个位置，使用rnn和交叉熵。
找到无法识别的字符(除了英文字符外的所有字符 所在的列(类别))
文本长度6(6个类别) 向量长度5

"""

class TorchModel(nn.Module):
    def __init__(self, vector_len, text_len, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_len, padding_idx=0)  #字符向量化
        self.rnn = nn.RNN(vector_len, 32, batch_first=True) # 增加hidden_size到32
        self.classifier = nn.Linear(32, text_len)  # 添加线性层，输出维度为text_len（6个位置）
        self.loss = nn.CrossEntropyLoss()  # 使用CrossEntropyLoss类

    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, text_len, vector_len)
        rnn_out, _ = self.rnn(x)                  #(batch_size, text_len, hidden_size)
        # 取RNN最后一个时间步的输出
        last_output = rnn_out[:, -1, :]           #(batch_size, hidden_size)
        y_pred = self.classifier(last_output)      #(batch_size, text_len)
        
        if y == None:
            return torch.softmax(y_pred, dim=1)
        else:
            return self.loss(y_pred, y)


def main():
    #配置参数
    epoch_num = 20        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.005 #学习率

    # 加载词表
    vocab = load_vocab()
    # 建立模型
    model = TorchModel(char_dim, sentence_length, vocab)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 加载训练样本
    X,Y = build_dataset(train_sample)

    # 训练过程
    for epoch in range(epoch_num):
        model.train() # 训练模式
        watch_loss = []
        for batch in range(train_sample // batch_size):
            start = batch * batch_size
            end = (batch + 1) * batch_size
            batch_x = X[start:end]
            batch_y = Y[start:end]
            loss = model.forward(batch_x, batch_y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            optim.zero_grad()    #梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        log.append(np.mean(watch_loss))
    #画图
    plt.plot(range(len(log)), log, label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'model.bin'))
    return

#使用训练好的模型做预测
def predict(model_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = load_vocab() #加载字符表
    model = TorchModel(char_dim, sentence_length, vocab)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        # 添加对未知字符的处理，使用['unk']作为未知字符的索引
        x.append([vocab.get(char, vocab['unk']) for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        # 获取最大概率的位置索引
        pred_pos = torch.argmax(result[i]).item()
        # 获取该位置的概率值
        prob = result[i][pred_pos].item()
        print("输入：%s, 预测位置：%d, 概率值：%.4f" % (input_string, pred_pos, prob))



if __name__ == "__main__":
    main()
    test_strings = ["=rtftt", "w+zdfg", "rq-deg", "xkw*ww", "ites/b", "jflka泥"]
    predict(os.path.join(os.path.dirname(__file__), 'model.bin'), test_strings)
