# coding:utf8
import random
import torch
import torch.nn as nn
import numpy as np
import json

"""
输入一个字符串，预测出指定字符的位置
"""

class week3Model(nn.Module):
    def __init__(self, vec_dim, sentence_length, vocab):
        super(week3Model, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vec_dim)
        self.rnn = nn.RNN(vec_dim, vec_dim, batch_first=True)
        self.classify = nn.Linear(vec_dim, sentence_length + 1)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y_true=None):
        x = self.embedding(x)
        rnn_out, hidden = self.rnn(x)
        x = rnn_out[:, -1, :]
        y_pred = self.classify(x)
        if y_true is not None:
            return self.loss(y_pred, y_true)
        else:
            return y_pred

def get_vocab():
    chars = "本周是nlp学习的第三周abcdefghijk"
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1
    vocab['unk'] = len(vocab)
    return vocab


def get_sample(vocab, sentence_length):
    x = random.sample(list(vocab.keys()), sentence_length)
    if "n" in x:
        y = x.index("n")
    else:
        y = sentence_length
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

def get_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = get_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_model(vocab, char_dim, sentence_length):
    model = week3Model(char_dim, sentence_length, vocab)
    return model


def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = get_dataset(500, vocab, sample_length)   #建立200个用于测试的样本
    print("本轮测试共%d个样本"%(len(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("本轮预测正确个数：%d, 当前正确率为：%f"%(correct, correct/(correct+wrong)))
    return


def main():
    #配置参数
    epoch_num = 50       #训练轮数
    batch_size = 50       #每次训练样本个数
    train_sample = 5000    #每轮训练总共训练的样本总数
    char_dim = 30         #每个字的维度
    sentence_length = 10   #样本文本长度
    learning_rate = 0.001 #学习率
    # 建立字表
    vocab = get_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        each_loss = []
        for i in range(int(train_sample / batch_size)):
            x, y = get_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            each_loss.append(loss.item())
        print("******当前是第%d轮，loss为: %f" % (epoch + 1, np.mean(each_loss)))
        acc = evaluate(model, vocab, sentence_length)
    #保存模型
    torch.save(model.state_dict(), "week3.model")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


def predict(model_path, vocab_path, input_strings):
    char_dim = 30
    sentence_length = 10  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    print('x是： %s' % x)
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
        print('模型forward的结果是：%s '% result)
    for i, input_string in enumerate(input_strings):
        print("输入字符串：%s, 预测结果：%s, 概率值：%s" % (input_string, torch.argmax(result[i]), result[i])) #打印结果



if __name__ == "__main__":
    main()
    test_strings = ['bdfacbdghn','dnbdghhabc','nbdghhabcl','本周cdenfghi']
    predict("week3.model", "vocab.json", test_strings)
