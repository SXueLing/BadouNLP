#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

from 赵康乾.week02.TorchDemo_MultiClassification import accuracy

"""

尝试修改nlpdemo，做一个多分类任务，判断特定字符在字符串的第几个位置，使用rnn和交叉熵。

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding Layer
        self.rnn = nn.LSTM(input_size=vector_dim, hidden_size=vector_dim, batch_first=True)  # LSTM Layer
        self.classify = nn.Linear(vector_dim, sentence_length + 1)  # Linear layer for classification
        self.loss = nn.CrossEntropyLoss()  # Cross-entropy loss function

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x, _ = self.rnn(x)                         # (batch_size, sen_len, vector_dim)
        x = x[:, -1, :]                            # Take the output of the last time step (batch_size, vector_dim)                        #(batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        x = self.classify(x)                       # (batch_size, vector_dim) -> (batch_size, sentence_length+1)
        if y is not None:
            return self.loss(x, y)                 #compute loss
        else:
            return torch.argmax(x, dim=1)          # Return predicted class indices

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
def build_sample(vocab, sentence_length, target_char="你"):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if target_char in x:
        y = x.index(target_char)  # Position of the target character
    else:
        y = sentence_length  # Class representing "not found"
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length, target_char="你"):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length, target_char)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sentence_length, target_char="你"):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length, target_char)   #建立200个用于测试的样本
    correct = 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        correct = (y_pred == y).sum().item()  #计算预测正确的个数
    accuracy = correct / len(y)  #计算准确率
    print(f"准确率: {accuracy:.2f}")
    return accuracy


def main():
    #配置参数
    epoch_num = 10        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.005 #学习率
    target_char = "你"  # Character to detect
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
            x, y = build_dataset(batch_size, vocab, sentence_length, target_char) #构造一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length, target_char)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="Accuracy")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="Loss")  #画loss曲线
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
def predict(model_path, vocab_path, input_strings, target_char="你"):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    model.eval()

    x = []
    for input_string in input_strings:
        # Pad or truncate each input string
        input_string = input_string[:sentence_length]  # Truncate if too long
        padded_string = input_string + " " * (sentence_length - len(input_string))
        x.append([vocab.get(char, vocab['unk']) for char in padded_string])  # 序列化输入

    x = torch.LongTensor(x)

    with torch.no_grad():  #不计算梯度
        predictions = model(x)  #模型预测
    for i, input_string in enumerate(input_strings):
        predicted_class = predictions[i].item()
        if predicted_class == sentence_length:
            print(f"Input: {input_string}, Prediction: '{target_char}' not found")
        else:
            print(f"Input: {input_string}, Prediction: '{target_char}' found at position {predicted_class}")

if __name__ == "__main__":
    main()
    test_strings = ["abcd你", "我abcdef", "xyzuvw", "defgh我"]
    predict("model.pth", "vocab.json", test_strings)
