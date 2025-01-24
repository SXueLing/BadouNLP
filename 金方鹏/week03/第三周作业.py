#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现
作业： 判断“你”出现的位置
"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding层
        # len(vocab)：词汇表的大小（即输入索引的范围，通常是类别数量）。
        # vector_dim：每个嵌入向量的维度。
        # padding_idx：可选参数，指定需要填充的索引（如0），使该索引始终映射为全零向量。
        # self.pool = nn.AvgPool1d(sentence_length)   #池化层
        self.layer = nn.RNN(vector_dim, hidden_size, bias=False, batch_first=True) #Rnn
        self.classify = nn.Linear(hidden_size, 7)     #全连接层
        # self.activation = torch.sigmoid     #sigmoid归一化函数

        self.loss = nn.CrossEntropyLoss()  #loss函数采用交叉熵

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        #print(x)
        x = self.embedding(x)
        #print(x)#(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        output, hidden = self.layer(x)  # RNN 输出 output：(batch_size, sen_len, hidden_size)
                                        # hidden：(num_layers, batch_size, hidden_size)
        #print(output.shape, hidden.shape)
        # 取最后一个时间步的隐藏状态作为特征
        last_hidden = hidden[-1]  # (batch_size, hidden_size)
        y_pred = self.classify(last_hidden)  # 通过全连接层分类
        if y is not None:
            #print(self.loss(y_pred, y))
            return self.loss(y_pred, y)#预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果

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
    # 判断字符出现位置
    if "你" in x:  # 出现返回位置 从1开始
        y = x.index("你")+1
    else:
        y = 0  # 未出现 类别0
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
def build_model(vocab, char_dim, sentence_length, hidden_size):
    model = TorchModel(char_dim, sentence_length, vocab, hidden_size)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    #print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), 200 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        y_pred_class = torch.argmax(y_pred, dim=1) #预测类别
        # print(x)
        # print(y)
        # print(y_pred_class)
        for y_p, y_t in zip(y_pred_class, y):  #与真实标签进行对比
            if y_p.item() == y_t.item():
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    #配置参数
    epoch_num = 10        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.005 #学习率
    # input_size = 6
    hidden_size = 128  #隐藏层
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length, hidden_size)
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
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    hidden_size = 128
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length, hidden_size)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
        result_class = torch.argmax(result, dim=1) #按行提取类别
    for i, input_string in enumerate(input_strings):
        #print(input_string)
        print(f"输入：{input_string}, 预测类别：{result_class[i].item()}, 概率值：{result[i].tolist()}")
        #print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, result_class[i].item(), result[i].tolist())) #打印结果


if __name__ == "__main__":
    #main()
    #vacub = build_vocab()
    #print(build_dataset(10,vacub,6))

    test_strings = ["f你vfee", "wz你dfg", "rqwde你", "n我kwww"]
    predict("model.pth", "vocab.json", test_strings)
