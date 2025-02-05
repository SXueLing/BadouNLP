
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import json

"""
实现一个简单的NLP任务：
找出英文文本中是否带有中文字符，并返回第一个中文字符的位置
"""

class DemoNLP(nn.Module):
    def __init__(self, vocab_size, char_vec_dim, output_dim):
        """
        in_args
        vocab_size 字典的大小
        char_vec_dim  字典中字符的向量维度
        output_dim  输出结果的维度
        """
        super(DemoNLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, char_vec_dim)
        self.rnn = nn.RNN(char_vec_dim, char_vec_dim, batch_first=True)
        self.linear = nn.Linear(char_vec_dim, output_dim) # 输出维度要包含没有中文字的情况，所以该值为sentence_length + 1
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)  # embedding曾  batch_size * sen_len  ->  batch_size * sen_len d* char_vec_dim
        output, hidden = self.rnn(x)  # output: batch_size * sen_len * char_vec_dim; hidden: 1 * sen_len * char_vec_dim
        hidden.squeeze_()  # 去除值为1的向量轴 。如果不去掉batch维度，无法和正确值的向量维度对齐，会报错 sen_len * char_vec_dim
        # print(hidden.shape)
        y_pred = self.linear(hidden)  #线性曾 sen_len * output_dim
        # print(y_pred.shape)
        # print(y.shape)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def generate_vocab():
    """
    将一个字符串生成一个字符集，类型是字典类型
    out_arg
    vocab  返回一个字符表，类型为字典类型
    """
    default_str = "abcde你我他一起跑ijklmnopqrstuvwxyz"
    vocab = {"pad":0}
    for index, char in enumerate(default_str):
        vocab[char] = index + 1  # 从数字1开始
    vocab["unk"] = len(vocab)  # unk代表超出字典的键值，默认用unk表示
    return vocab

def generate_sample(vocab, sentence_length):
    """
    1 从字符集中随机生成指定长度的文本
    2 返回这段文本和第一个中文你我他的位置
    in_args
    vocab  字符表
    sentence_length  语句长度
    out_args
    1 如果存在中文，返回对应语句序列和对应位置索引
    2 如果不存在中文， 返回度地应语句序列和语句长度
    """
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]  # 随机从字符表中生成sentence_len长度个字符
    chinese_chars = ['你', '我', '他','一', '起', '跑'] # 中文过滤集合
    for index, char in enumerate(x):  # 提取sentence中的字符和索引信息
        if char in chinese_chars: # 查找每个字符中是否存在中文
            # print(x, end="\n")
            return [vocab.get(word, vocab['unk']) for word in x], index #若存在中文，返回对应的语句序列和对应的index（0～4）

    return [vocab.get(word, vocab['unk']) for word in x], len(x) # 若无，则也返回对应的语句序列，返回一个标记5

def generate_dataset(samples_num, vocab, sentence_length):
    """
    生成数据集
    in_args
    samples_num  样本总数量
    vocab  字符表
    sentence_length  语句长度，即语句中字符个数
    out_arg
    1 按照字符表生成的语句向量
    2 对应1中，中文字符在语句中所在的位置索引
    """
    X = []
    Y = []
    for _ in range(samples_num):
        x, y = generate_sample(vocab, sentence_length)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X), torch.LongTensor(Y) # 返回tensor张量格式

def evaluate(model, vocab, samples_num, sentence_length):
    """
    每轮训练之后，生成batch_size个样本进行预测，计算正确率并返回
    in_args
    model  训练完的模型
    vocab  字符表
    samples_num  评估所用的样本总数
    sentence_length  语句的长度，即组成该语句的字符个数
    out_arg
    返回正确率
    """
    model.eval() # 切换为评估模式
    x, y = generate_dataset(samples_num, vocab, sentence_length)  # 生成；数据集
    correct, wrong = 0, 0
    with torch.no_grad(): #不计算梯度
        y_pred = model(x) #只传入x，不传入正确值，使得模型返回的时预测值
        for y_p, y_t in zip(y_pred, y):
            if int(torch.argmax(y_p)) == int(y_t): # 如果预测生成的向量中，概率v最大的值的下标和正确值一样，说明预测正确
                correct += 1
            else:
                wrong += 1

    print("%d 个样本， 正确率为: %f" % (samples_num, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    """
    训练的主程序
    """
    epoch_num = 20  #训练轮数
    batch_size = 20 #一次训练的样本个数
    train_samples = 4000 #训练样本的总个数
    char_dim = 20 #字符的向量维度，用于embedding
    sentence_length = 5 #语句的长度，即组成语句的字符个数
    learning_rate = 0.001 #学习率

    vocab = generate_vocab()  # 生成字符表
    model = DemoNLP(len(vocab), char_dim, sentence_length + 1) #生成模型。注意最后一个参数，要考虑没有中文时的语句。相当于变成了6选1
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # 构建优化器
    log = []
    for epoch in range(epoch_num):
        model.train() # 开启训练模式。如果有pooling曾的话就会生效
        watch_loss = []
        for batch in range(int(train_samples / batch_size)): # 一次训练batch个样本，训练完一轮需要多少次
            x, y = generate_dataset(batch_size, vocab, sentence_length) # 生成样本
            optimizer.zero_grad() # 梯度归零
            loss = model(x, y) #计算loss
            loss.backward() #反向传播，即计算梯度
            optimizer.step() # 更新权重
            watch_loss.append(loss.item()) # 记录bathc——size个样本训练之后的loss值
        print("----第%d轮平均loss: %f----\n" % (epoch + 1, np.mean(watch_loss))) # 本轮训练的loss求一次平均
        acc = evaluate(model, vocab, int(train_samples / batch_size), sentence_length) # 评估本次训练的正确率
        log.append([acc, np.mean(watch_loss)])  #将本轮训练的正确率和loss值保存

    plt.plot(range(len(log)), [l[0] for l in log], label="acc") # 横轴为epoch的轮数，纵轴为正确率
    plt.plot(range(len(log)), [l[1] for l in log], label="loss") # 横轴为epoch的轮数， 纵轴为loss值
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), "NLPmodel.pth") # 保存模型

    writer = open("NLPvocab.json", "w", encoding="utf-8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2)) #保存字符表
    writer.close()
    return

def predict(model_path, vocab_path, test_samples):
    """
    预测函数，对输入的给定字符串进行分类 ，输出类型范围0 ～ 4分别对应中文在字符串中的位置，5代表无中文
    in_args
    model_path  模型的位置路经
    vocab_path  字符表的位置路经
    test_samples  测试集
    """
    char_dim = 20  # 测试语句中字符的维度，和训练是保持一致
    sentence_length = 5  # 语句的长度，即一句话包含几个字符。和训练是保持一致
    vocab = json.load(open(vocab_path, "r", encoding="utf-8"))  # 加载字符表
    model = DemoNLP(len(vocab), char_dim, sentence_length + 1)
    model.load_state_dict(torch.load(model_path)) #加载训练好的模型
    x = []
    for test_sample in test_samples:
        # x.append([vocab[char] for char in test_sample]) # 做成batch形状，即 batch_size * sen_len
        # 做成batch形状，即 batch_size * sen_len. 这里构建sentence时使用unk，测试集中使用了f这个字符表中没有的字符
        x.append([vocab.get(char, vocab['unk']) for char in test_sample])
    model.eval() #
    with torch.no_grad(): #不计算梯度
        result = model.forward(torch.LongTensor(x))  #等到预测结果
    for index, test_sample in enumerate(test_samples):
        print("sample_str: %s, 预测类别: %s, 概率值: %s" % (test_sample, torch.argmax(result[index]), result[index]))

if __name__ == '__main__':
    main()
    test_samples = ["我addo", "bcdte", "aci你我", "f一起de", "iopuy"]  #注意，字符一定是字符表中定义的字符
    predict("NLPmodel.pth", "NLPvocab.json", test_samples)


# dict = generate_vocab()
# # str, index = generate_sample(dict, 5)
# str, index = generate_dataset(3, dict, 5)
# print(str, index)
