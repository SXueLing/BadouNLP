
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

"""

class TorchModel(nn.Module):
    def __init__(self, char_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), char_dim, padding_idx=0)  #embedding层
        # rnn 的 input_size 需要和样本文本长度保持一致
        self.layer = nn.RNN(char_dim, 26, bias=False, batch_first=True)  #隐含层
        # 线性层的第一个参数需要和 rnn 中的 hidden_size 保持一致
        # 线性层第二个参数需要比build_sample阶段y的最大值大1
        self.linear = nn.Linear(26, sentence_length + 1)     #线性层
        self.loss = nn.functional.cross_entropy   #loss函数采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)      #(batch_size, sen_len) -> (batch_size, sen_len, char_dim)
        # output : (batch_size, sen_len, char_dim)
        # hidden : (1, char_dim, hidden_size)
        output, hidden = self.layer(x)
        # 使用最后一个隐藏状态作为分类的输入
        # output[:, -1, :] 和 hidden 值一致，但是拿到的值是会多一维，需要将第0维降维之后才能传到线性层
        # y_pred = self.linear(output[:, -1, :].squeeze(0))
        y_pred = self.linear(hidden.squeeze(0))  #(1, char_dim, hidden_size) -> (char_dim, sentence_length + 1)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #27
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
def build_sample(vocab, sentence_length, find_char):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if find_char in x:
        y = x.index(find_char)
    else:
        y = sentence_length
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, sentence_length, vocab, find_char):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length, find_char)
        dataset_x.append(x)
        # 使用交叉熵损失函数时，这里的 y 不需要转成 [y], 交叉熵计算时需要的是具体的数字，而不是列表，加上会报错
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


#建立模型
def build_model(vocab, sentence_length, char_dim):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sentence_len, find_char):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_len, find_char)   #建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            # 最大值下标与真实值相等时，则为预测正确
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main(find_char):
    print("需要训练的字符为：{}".format(find_char))
    #配置参数
    epoch_num = 20        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005 #学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, sentence_length, char_dim)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, sentence_length, vocab, find_char) #构造一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, sentence_length, vocab, find_char)   #测试本轮模型结果
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
def predict(model_path, vocab_path, input_strings, find_char):
    print("\n需要查找的字符为：{}".format(find_char))
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, sentence_length, char_dim)     #建立模型，这一步传入的参数需要和模型训练时传入的参数保持一致
    model.load_state_dict(torch.load(model_path, weights_only=True))             #加载训练好的权重, weights_only默认为false，不置为true会有告警出现
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab['unk']) for char in input_string])
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        if result[i].argmax() < sentence_length:
            # 取出最大值下标加上1即为预测的类型
            index = int(torch.argmax(result[i]) + 1)
        else:
            index = -1
        print("输入：%s, 预测类别：%s" % (input_string, index)) #打印结果


if __name__ == "__main__":
    # 需要训练和查找的特定字符
    find_char = "f"
    main(find_char)
    # test_strings = ["fnvfee", "wz你dfg", "rqwdeg", "n我kwww"]
    test_strings = ["anvfee", "wzaffg", "rqwdeg", "nfkwww"]
    # # test_strings = ["d", "e", "f", "我"]
    predict("model.pth", "vocab.json", test_strings, find_char)
