#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个多分类任务，判断特定字符在字符串的第几个位置，使用rnn和交叉熵

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding 层
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)  # 添加 RNN 层
        self.classify = nn.Linear(vector_dim, sentence_length)  # 线性层，输出为句子长度大小的类别
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失

    # 当输入真实标签，返回 loss 值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x, _ = self.rnn(x)  # (batch_size, sen_len, vector_dim)
        x = x[:, -1, :]  # 取 RNN 最后一层输出 (batch_size, vector_dim)
        x = self.classify(x)  # (batch_size, vector_dim) -> (batch_size, sentence_length)
        if y is not None:
            return self.loss(x, y)  # 计算交叉熵损失
        else:
            return x  # 输出预测结果

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
def build_sample(vocab, sentence_length):
    # 随机生成一个句子
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 随机选定目标字符的索引
    target_idx = random.randint(0, sentence_length - 1)
    target_char = random.choice("你我他")  # 特定字符
    x[target_idx] = target_char  # 将目标字符放入随机位置
    x = [vocab.get(word, vocab['unk']) for word in x]  # 转换成序号
    return x, target_idx  # 返回输入序列和目标索引

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)  # y 类型改为 LongTensor

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 生成测试样本
    correct, total = 0, len(y)
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        y_pred = torch.argmax(y_pred, dim=1)  # 获取预测的类别索引
        correct = (y_pred == y).sum().item()  # 统计正确预测数量
    accuracy = correct / total
    print(f"正确预测个数：{correct}, 总数：{total}, 准确率：{accuracy:.4f}")
    return accuracy


def main():
    #配置参数
    epoch_num = 10        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.005 #学习率
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
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载训练好的权重
    x = [[vocab.get(char, vocab['unk']) for char in string] for string in input_strings]  # 序列化输入
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        result = model(torch.LongTensor(x))  # 模型预测
        predictions = torch.argmax(result, dim=1)  # 获取预测结果
    for i, input_string in enumerate(input_strings):
        print(f"输入：{input_string}, 预测位置：{predictions[i].item()}")



if __name__ == "__main__":
    main()
    test_strings = ["fnvfe他", "wz你dfg", "他rqwde", "n我kwww"]
    predict("model.pth", "vocab.json", test_strings)
