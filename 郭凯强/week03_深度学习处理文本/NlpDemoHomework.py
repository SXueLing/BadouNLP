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
判断特定字符在字符串的第几个位置，使用rnn和交叉熵

"""
class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # Embedding 层
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)  # RNN 层
        self.classify = nn.Linear(vector_dim, 1)  # 每个时间步的分类层
        self.activation = torch.sigmoid  # Sigmoid 激活函数
        self.loss = nn.functional.cross_entropy  # 使用交叉熵损失函数

    def forward(self, x, y=None):
        """
        x: 输入张量，形状为 (batch_size, sentence_length)
        y: 标签张量，形状为 (batch_size, sentence_length)
        """
        x = self.embedding(x)  # (batch_size, sentence_length) -> (batch_size, sentence_length, vector_dim)
        x, _ = self.rnn(x)  # RNN 输出 (batch_size, sentence_length, vector_dim)
        x = self.classify(x)  # 每个时间步的线性层 (batch_size, sentence_length, vector_dim) -> (batch_size, sentence_length, 1)
        y_pred = self.activation(x).squeeze(-1)  # Sigmoid 激活 -> (batch_size, sentence_length)

        if y is not None:
            # 计算损失，将预测值 y_pred 和标签 y 输入损失函数
            return self.loss(y_pred, y)
        else:
            # 返回预测值
            return y_pred

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
def build_sample(vocab, sentence_length, target_char):
    # 随机从字表中选取 sentence_length 个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    target_index = random.randint(0, sentence_length - 1)
    x = x[:target_index] + [target_char] + x[target_index+1:]
    # 初始化标签 y，长度与 x 相同，默认值为 0
    y = [0] * sentence_length

    # 如果当前字符是目标字符，则标记位置索引为正
    for i, char in enumerate(x):
        if char == target_char:
            y[i] = 1  # 标签值为 位置索引（从 1 开始）

    # 将字符转换成索引，为了后续的 embedding 操作
    x = [vocab.get(word, vocab['unk']) for word in x]

    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length, target_char):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length, target_char)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length, target_char):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length, target_char)   #建立200个用于测试的样本
    print("本次预测集中共有%d个样本"%(len(x)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if np.argmax(y_p) == np.argmax(y_t):
                correct += 1   #负样本判断正确
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
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    target_char = "你"
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

# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表

    # 加载模型（注意模型需匹配新的定义）
    model = TorchModel(char_dim, sentence_length, vocab)  # 替换为新的模型定义
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    # 对输入数据进行处理
    x = []
    for input_string in input_strings:
        # 将输入序列化，如果长度不足则填充，超长则截断
        x_encoded = [vocab.get(char, vocab['unk']) for char in input_string]
        if len(x_encoded) < sentence_length:
            x_encoded += [vocab['unk']] * (sentence_length - len(x_encoded))  # 填充
        else:
            x_encoded = x_encoded[:sentence_length]  # 截断
        x.append(x_encoded)

    # 转为张量并预测
    x = torch.LongTensor(x)
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model(x)  # 模型预测，形状 (batch_size, sentence_length)

    # 逐字符打印预测结果
    for i, input_string in enumerate(input_strings):
        print(f"输入：{input_string}")
        for j, char in enumerate(input_string[:sentence_length]):  # 遍历有效字符
            prob = result[i][j].item()  # 当前字符的预测概率
            pred = 1 if prob >= 0.5 else 0  # 概率阈值判定
            print(f"  位置 {j + 1}: 字符 '{char}' -> 预测: {pred} (概率: {prob:.4f})")
        print()


if __name__ == "__main__":
    main()
    test_strings = ["fnvfee", "wz你dfg", "r你deg你", "n你kww你"]
    predict("model.pth", "vocab.json", test_strings)
"""
使用保存好的模型输出脚本如下，可以看到结果是正确的
输入：fnvfee
  位置 1: 字符 'f' -> 预测: 0 (概率: 0.0010)
  位置 2: 字符 'n' -> 预测: 0 (概率: 0.0006)
  位置 3: 字符 'v' -> 预测: 0 (概率: 0.0009)
  位置 4: 字符 'f' -> 预测: 0 (概率: 0.0007)
  位置 5: 字符 'e' -> 预测: 0 (概率: 0.0006)
  位置 6: 字符 'e' -> 预测: 0 (概率: 0.0007)

输入：wz你dfg
  位置 1: 字符 'w' -> 预测: 0 (概率: 0.0022)
  位置 2: 字符 'z' -> 预测: 0 (概率: 0.0005)
  位置 3: 字符 '你' -> 预测: 1 (概率: 0.9991)
  位置 4: 字符 'd' -> 预测: 0 (概率: 0.0014)
  位置 5: 字符 'f' -> 预测: 0 (概率: 0.0006)
  位置 6: 字符 'g' -> 预测: 0 (概率: 0.0012)

输入：r你deg你
  位置 1: 字符 'r' -> 预测: 0 (概率: 0.0026)
  位置 2: 字符 '你' -> 预测: 1 (概率: 0.9990)
  位置 3: 字符 'd' -> 预测: 0 (概率: 0.0013)
  位置 4: 字符 'e' -> 预测: 0 (概率: 0.0006)
  位置 5: 字符 'g' -> 预测: 0 (概率: 0.0014)
  位置 6: 字符 '你' -> 预测: 1 (概率: 0.9990)

输入：n你kww你
  位置 1: 字符 'n' -> 预测: 0 (概率: 0.0012)
  位置 2: 字符 '你' -> 预测: 1 (概率: 0.9991)
  位置 3: 字符 'k' -> 预测: 0 (概率: 0.0021)
  位置 4: 字符 'w' -> 预测: 0 (概率: 0.0010)
  位置 5: 字符 'w' -> 预测: 0 (概率: 0.0009)
  位置 6: 字符 '你' -> 预测: 1 (概率: 0.9990)
"""
