import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
    尝试修改nlpdemo，做一个多分类任务，判断特定字符在字符串的第几个位置，使用rnn和交叉熵。
"""
class RNNModel(nn.Module):
    def __init__(self, vocab, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), input_size, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(input_size, hidden_size, bias=False, batch_first=True)  # RNN
        self.classify = nn.Linear(hidden_size, output_size)  # 线性层
        self.loss = nn.functional.cross_entropy  # 损失函数

    def forward(self, x, y=None):
        x = self.embedding(x)
        output, h = self.rnn(x)  # 激活函数
        h = output[:, -1, :]
        h = self.classify(h)  # 线性变换
        if y is not None:
            return self.loss(h, y)  # 计算损失
        return h
# 字符集随便挑了一些字，实际上还可以补充
# 为每个字生成一个标号
def build_vocab():
    chars = "你我他"
    vocab = {"pad": 0}
    for i, char in enumerate(chars):
        vocab[char] = i + 1
    vocab["unk"] = len(vocab)
    return vocab

# 随机生成一个样本
# 从所有字中选取sentence_length个字
def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    target_char = random.choice("我")
    if target_char in x:
        position = x.index(target_char)
    else:
        position = 0
    x = [vocab.get(char, vocab["unk"]) for char in x]
    return x, position

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

# 建立模型
def build_model(vocab, input_size, hidden_size, output_size):
    model = RNNModel(vocab, input_size, hidden_size, output_size)
    return model

# 测试代码
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)
    current, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p) == y_t:
                current += 1
            else:
                wrong += 1
    print("准确率:%f" % (current / (current + wrong)))
    return current / (current + wrong)


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 5  # 每次训练样本个数
    train_sample = 500  # 每轮训练的样本总数
    sentence_length = 2  # 每个样本的长度
    input_size = 2  # 每个字符的维度
    hidden_size = 4  # 每个字符计算出来的维度
    learning_rate = 0.005  # 学习率
    # 构建词汇表
    vocab = build_vocab()
    # 构建模型
    model = build_model(vocab, input_size, hidden_size, sentence_length)
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample // batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # x: 5*2  y: 5*1
            optimizer.zero_grad()  # 梯度清零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
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
def predict(model, vocab_path, input_strings):
    sentence_length = 2  # 样本字符长度
    input_size = 2  # 每个字符的维度
    hidden_size = 4  # 每个字符计算出来的维度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载词表
    model = build_model(vocab, input_size, hidden_size, sentence_length)  # 建立模型
    model.load_state_dict(torch.load("model.pth"))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab["unk"]) for char in input_string])  # 将输入数字化
    model.eval()  # 测试模式
    with torch.no_grad():
        result = model(torch.LongTensor(x))  # 预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测的结果：%s, %s" % (input_string, torch.argmax(result[i]), result[i]))

if __name__ == "__main__":
    # main()
    test_strings = ["你我", "我你", "我啊", "它我"]
    predict("model.pth", "vocab.json", test_strings)
