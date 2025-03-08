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
判断文本中特定字符出现的位置

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)  # 使用 RNN 层
        self.classify = nn.Linear(vector_dim, sentence_length + 1)   # 输出维度为句子长度+1
        self.loss = nn.CrossEntropyLoss()                            # 使用交叉熵损失

    def forward(self, x, y=None):
        x = self.embedding(x)                    # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        rnn_output, _ = self.rnn(x)              # (batch_size, sen_len, vector_dim)
        rnn_output = rnn_output[:, -1, :]        # 取最后一个时间步的输出 (batch_size, vector_dim)
        y_pred = self.classify(rnn_output)       # (batch_size, sentence_length + 1)
        if y is not None:
            return self.loss(y_pred, y.squeeze().long())  # 计算损失
        else:
            return torch.argmax(y_pred, dim=1)   # 返回预测的位置索引

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
def build_sample(vocab, sentence_length,char_to_find):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    #如果包含字符‘你’,则返回所在位置，否则返回-1
    index = next((i for i, char in enumerate(x) if char == char_to_find), -1)
    y = index + 1
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length, char_to_find):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length, char_to_find )
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length, char_to_find):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length, char_to_find )   #建立200个用于测试的样本
    print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), 200 - sum(y)))
    correct,incorrect, wrong = 0, 0,0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if int(y_p) == int(y_t):
                correct += 1   #负样本判断正确
            elif int(y_p) != int(y_t) :
                incorrect += 1   #正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+incorrect+wrong)))
    return correct/(correct+incorrect+wrong)


def main():
    #配置参数
    epoch_num = 10        #训练轮数
    batch_size = 10       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.005 #学习率
    char_to_find = '你'   #要查找的字符
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
            x, y = build_dataset(batch_size, vocab, sentence_length,char_to_find) #构造一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length, char_to_find )   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    torch.save(model.state_dict(), "model2.pth")
    # 保存词表
    writer = open("vocab2.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    #return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    x = []
    print(input_strings)    
    for input_string in input_strings:
        #padded_input = list(input_string.ljust(sentence_length, 'pad')[:sentence_length])
        x.append([vocab[char] for char in input_string])
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    print("=result=result=============result=====================")    
    for i, input_string in enumerate(input_strings):
        position = result[i].item() - 1
        print(f"输入：{input_string}, 预测位置：{position if position >=0 else 'not found'}")


if __name__ == "__main__":
    main()
    input_strings = ["fnvfee", "wz你dfg", "n我kwww", "你我kwww", "我kwww你"]
    print("=开始predict===================================")
    print("=开始predict===================================")
    print("=开始predict===================================")
    print("=开始predict===================================")
    predict("model2.pth", "vocab2.json", input_strings)
