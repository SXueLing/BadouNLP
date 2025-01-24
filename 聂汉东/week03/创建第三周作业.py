import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt


# 定义RNN模型
class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_dim=64):
        super(TorchModel, self).__init__()
        self.vocab = vocab
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(vector_dim, hidden_dim, batch_first=True)  # RNN层
        self.fc = nn.Linear(hidden_dim, len(vocab))  # 修改全连接层输出维度为词表大小
        self.softmax = nn.Softmax(dim=-1)  # Softmax激活函数，用于分类
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    # 前向传播
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        out, _ = self.rnn(x)  # (batch_size, sen_len, hidden_dim)
        out = out[:, -1, :]  # 取最后一个时间步的输出， (batch_size, hidden_dim)
        out = self.fc(out)  # (batch_size, len(vocab)) 修改为词表大小
        out = self.softmax(out)  # 使用Softmax进行多分类

        if y is not None:
            return self.loss(out, y.view(-1))  # 计算交叉熵损失
        else:
            return out  # 返回预测值


# 构建字表
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # unk字符
    return vocab


# 构建样本
def build_sample(vocab, sentence_length, target_char="你"):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]

    # 查找目标字符位置
    if target_char in x:
        y = x.index(target_char) + 1  # 字符位置（1-based）
    else:
        y = 0  # 字符没有出现，返回0

    # 将字转换成序号，为了做embedding
    x = [vocab.get(word, vocab['unk']) for word in x]

    return x, y


# 构建数据集
def build_dataset(sample_length, vocab, sentence_length, target_char="你"):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length, target_char)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


# 测试代码
def evaluate(model, vocab, sample_length, sentence_length, target_char="你"):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length, target_char)  # 建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            predicted = y_p.argmax().item()  # 获取预测的类别
            if predicted == y_t.item():
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率
    target_char = "你"  # 目标字符，判断它在文本中的位置

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
            x, y = build_dataset(batch_size, vocab, sentence_length, target_char)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length, sentence_length, target_char)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), "model.pth")

    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()

    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings, target_char="你"):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab['unk']) for char in input_string])  # 将输入序列化，处理未知字符

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测

    for i, input_string in enumerate(input_strings):
        predicted_position = result[i].argmax().item()  # 获取预测的字符位置
        print("输入：%s, 预测字符位置：%d, 概率值：%f" % (
        input_string, predicted_position, result[i][predicted_position]))  # 打印结果


if __name__ == "__main__":
    main()
    test_strings = ["fnvfee", "wz你dfg", "rqwdeg", "n我kwww"]
    predict("model.pth", "vocab.json", test_strings)
