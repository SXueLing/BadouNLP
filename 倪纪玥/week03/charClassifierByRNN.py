import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

 尝试修改nlpdemo，做一个多分类任务，判断特定字符在字符串的第几个位置，使用rnn和交叉熵。
 NLP任务改造成多分类任务，”你“在这句话哪个位置 就是第几类  RNN 加 交叉熵  你不出现 你在哪个位置

输入：你nvfee,  预测类别：0, 概率分布：[8.6315608e-01 2.0123541e-04 6.2265560e-02 3.0513966e-04 5.0314978e-02 8.6003449e-03 1.5156469e-02]
输入：w你zdfg,  预测类别：1, 概率分布：[1.19179465e-04 9.06015396e-01 8.02825962e-04 6.01034351e-02 1.43966777e-03 6.88983081e-03 2.46296879e-02]
输入：rq你wde,  预测类别：2, 概率分布：[2.6599067e-01 6.3092937e-04 7.0110357e-01 1.8048103e-03 8.8202357e-03 5.4422668e-03 1.6207576e-02]
输入：n我k你ww, 预测类别：3, 概率分布：[3.1263469e-04 6.6465564e-02 1.8047835e-03 9.1890949e-01 1.6913754e-03 2.1816783e-03 8.6346278e-03]
输入：wzdf你x,  预测类别：4, 概率分布：[5.4811373e-02 5.3860096e-04 1.7901780e-02 5.6378025e-04 9.0844554e-01 1.1063528e-03 1.6632637e-02]
输入：zzdfx你,  预测类别：5, 概率分布：[2.7939351e-03 2.0057404e-02 2.6677758e-03 2.8915438e-03 8.8831177e-04 9.3587303e-01 3.4828097e-02]
输入：qzdfxx,   预测类别：6, 概率分布：[1.2727266e-03 9.6583966e-04 2.7949829e-04 1.5803752e-04 1.9160053e-04 4.8094837e-04 9.9665135e-01]

"""


class TorchRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab, output_size):
        super(TorchRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_size, bias=True, batch_first=True)  # RNN 层
        self.fc = nn.Linear(hidden_size, output_size)  # 将 RNN 输出映射到分类空间。
        # 移除激活函数，直接使用线性输出（因为 CrossEntropyLoss 交叉熵损失在内部会执行 Softmax 操作）
        self.loss = nn.CrossEntropyLoss()  # 损失函数为交叉熵

    def forward(self, x, y=None):
        x = self.embedding(x)   # [batch_size, seq_length, embedding_dim]
        out, hn = self.rnn(x)   # [batch_size, seq_length, hidden_size]
        out = out[:, -1, :]     # 取最后一个时间步的输出
        y_pred = self.fc(out)   # [batch_size, output_size]

        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 字符集随便挑了一些字，实际上还可以扩充
# 为每个字生成一个标号
# {"a":1, "b":2, "c":3...}
# abc -> [1,2,3]
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}  # 填充字符（padding）通常用于对文本序列进行补齐
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab["unk"] = len(vocab)  # 27 = 1 + 26 + 1 -1
    print(vocab)
    return vocab


# 随机生成一个样本
# 从所有字中选取 sentence_length 可重复的字符
# 判断生成的样本中 bingo 是第一次出现的位置
def build_sample(vocab, sentence_length, bingo):
    # 从字符集中生成 sentence_length 个可能重复的字符
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]  # 列表推导式

    # 判断 bingo 是否在样本中
    if bingo in x:
        # 返回 bingo 第一次出现的位置（从 0 开始计数）
        bingo_position = x.index(bingo)
    else:
        # 如果 bingo 不存在，返回 sentence_length 表示未出现
        bingo_position = sentence_length

    # 将字转换成序号，为了做embedding
    # 如果 word 不在 vocab（未知字符），则返回默认值 vocab['unk']
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, bingo_position


# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length, bingo):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length, bingo)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.tensor(dataset_x), torch.tensor(dataset_y)


# 建立模型
def build_model(embedding_dim, hidden_size, vocab_size, output_size):
    model = TorchRNN(embedding_dim, hidden_size, vocab_size, output_size)
    return model


#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sentence_length, bingo):
    model.eval()
    test_sample_num = 500
    x, y = build_dataset(test_sample_num, vocab, sentence_length, bingo)  #建立200个用于测试的样本
    print("本次预测集中共有%d个类别样本" % test_sample_num)
    with torch.no_grad():
        y_pred_logits = model(x)  # 获取模型的输出（logits）
        y_pred = torch.argmax(y_pred_logits, dim=1)  # 获取预测的类别索引
        correct = (y_pred == y).sum().item()  # 计算预测正确的数量
    total = y.size(0)
    accuracy = correct / total
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy


def main():
    # 配置参数
    bingo = "你"
    epoch_num = 10      # 训练轮数
    batch_size = 20     # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    embedding_dim = 20  # 每个字的维度 input_size
    hidden_size = 32    # 隐藏层的维度 hidden_size
    sentence_length = 6  # 样本文本长度
    output_size = sentence_length + 1  # 输出类别数
    learning_rate = 0.005  # 学习率

    # 建立字表
    vocab = build_vocab()

    # 建立模型
    model = build_model(embedding_dim, hidden_size, vocab, output_size)

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(train_sample // batch_size):
            x, y = build_dataset(batch_size, vocab, sentence_length, bingo)
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length, bingo)  #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
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
def predict(model_path, vocab_path, input_strings):
    embedding_dim = 20  # 每个字的维度 input_size
    hidden_size = 32    # 隐藏层的维度 hidden_size
    sentence_length = 6  # 样本文本长度
    output_size = sentence_length + 1  # 输出类别数
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    print("x.len = %d" % len(x))

    model = build_model(embedding_dim, hidden_size, vocab, output_size)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    model.eval()  # 测试模式

    with torch.no_grad():  # 不计算梯度
        output_logits = model.forward(torch.tensor(x))  # 模型预测
        predicted_classes = torch.argmax(output_logits, dim=1)  # 获取预测的类别索引
        probabilities = torch.softmax(output_logits, dim=1)  # 将 logits 转化为概率分布

    # 打印结果
    for vec, pred_class, prob in zip(input_strings, predicted_classes, probabilities):
        print("输入：%s, 预测类别：%d, 概率分布：%s" % (vec, pred_class.item(), prob.numpy()))


if __name__ == "__main__":
    main()
    test_strings = ["你nvfee", "w你zdfg", "rq你wde", "n我k你ww", "wzdf你x", "zzdfx你", "qzdfxx"]
    predict("model.pth", "vocab.json", test_strings)
