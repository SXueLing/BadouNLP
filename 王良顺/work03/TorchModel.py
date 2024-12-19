# coding:utf8
# auther: 王良顺

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import DataParallel

import BuildDataset

"""
将分类任务改造为多分类任务，找一个特定字符所在的位置，特定字符位置在第几位，就是第几类
"""

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载字表
vocab = json.load(open("vocab.json", "r", encoding="utf8"))

class TorchModel(nn.Module):
    def __init__(self, vector_dim, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding层
        self.dp = torch.nn.Dropout(0.5) # Dropout 层,丢失概率0.5
        self.rnn = nn.RNN(vector_dim, vector_dim, bias=False, batch_first=True) # input_size = 20 hidden_size=20
        self.liner1 = nn.Linear(vector_dim, 256)
        self.liner2 = nn.Linear(256, 11)    # 线性层 10个字符,分类任务有11类,最终输出该任务属于该类的概率值
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                    #(batch_size, sen_len) 20,10 -> (batch_size, sen_len, vector_dim) 20,10,20
        output, x = self.rnn(x)                  #(batch_size, sen_len, vector_dim) 20,10,20 ->(batch_size, sen_len, vector_dim) 20,10,20
        x = x.squeeze(0)                         #对第一维降维 torch.Size([20, 20])
        x = self.liner1(x)                       #(batch_size, vector_dim) -> (batch_size, 1) 20*20 20*256 -> 20*256
        x = self.dp(x)                           # 随机丢弃一些点
        y_pred = self.liner2(x)                  # 20*256 256*11 -> 20*11
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = BuildDataset.build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if int(y_t) == y_p.argmax():
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def main():
    #配置参数
    epoch_num = 20        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 2000    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 10   #样本文本长度
    learning_rate = 0.001 #学习率

    # 建立模型
    model = TorchModel(char_dim, vocab)
    # 将模型包装在DataParallel中
    model = DataParallel(model)
    # 将模型移动到GPU上
    model = model.cuda()
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = BuildDataset.build_dataset(batch_size, vocab, sentence_length) #构造一组训练样本
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
    torch.save(model.state_dict(), "model.pt")
    return

if __name__ == "__main__":
    main()
