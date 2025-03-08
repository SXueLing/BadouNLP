import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch的网络编写
实现一个多分类任务，判断特定字符在字符串中的位置
"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_dim=50):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(vector_dim, hidden_dim, batch_first=True)
        self.classify = nn.Linear(hidden_dim, sentence_length+1)
        self.loss = nn.CrossEntropyLoss(ignore_index=sentence_length)  # 使用交叉熵损失

    def forward(self, x, y=None, sentence_length=6):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        rnn_out, _ = self.rnn(x)  # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, hidden_dim)
        logits = self.classify(rnn_out[:, -1, :])
        if y is not None:
            return self.loss(logits, y)  # 返回交叉熵损失
        else:
            return logits

# 构建词汇表
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1   # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 未知字符
    return vocab

# 构建样本
def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    target_char = "你"  # 假设我们关注的字符是"你"
    # 判断目标字符的位置
    if target_char in x:
        y = x.index(target_char)  # 目标字符的索引位置
    else:
        y = sentence_length  # 如果没有目标字符，返回一个额外的类别（sentence_length表示无目标字符）
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号
    return x, torch.LongTensor([y])  # 标签为位置或无目标字符的标签

# 构建数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 构建模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

# 评估函数
def evaluate(model, vocab, sample_length, sentence_length):
    model.eval()
    x, y = build_dataset(sample_length, vocab, sentence_length)  # 构造测试样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for pred, true in zip(y_pred, y):  # 与真实标签进行对比
            predicted_position = torch.argmax(pred).item()  # 预测位置
            true_position = true.item()
            if predicted_position == true_position:
                correct += 1  # 预测正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    epoch_num = 10         # 训练轮数
    batch_size = 20        # 每次训练样本个数
    train_sample = 500     # 每轮训练总共训练的样本总数
    char_dim = 20          # 每个字的维度
    sentence_length = 6    # 样本文本长度
    learning_rate = 0.005  # 学习率
    vocab = build_vocab()  # 构建字表
    model = build_model(vocab, char_dim, sentence_length)  # 构建模型
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 选择优化器
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算损失
            loss.backward()     # 计算梯度
            optim.step()        # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, 200, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 绘制准确率和损失曲线
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

    # 保存模型和词汇表
    torch.save(model.state_dict(), "model.pth")
    with open("vocab.json", "w", encoding="utf8") as writer:
        json.dump(vocab, writer, ensure_ascii=False, indent=2)

def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 构建模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab['unk']) for char in input_string])  # 将输入序列化
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        predicted_position = torch.argmax(result[i]).item()  # 判断字符位置
        print(f"输入：{input_string}, 预测字符位置：{predicted_position}, 概率值：{torch.max(result[i]).item()}")

if __name__ == "__main__":
    main()
    test_strings = ["f你vfee", "wz你dfg", "rqwde你", "n你kwww"]
    predict("model.pth", "vocab.json", test_strings)
