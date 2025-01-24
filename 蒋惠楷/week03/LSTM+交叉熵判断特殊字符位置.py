import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

'''构建LSTM+交叉熵模型'''
class TorchLSTMModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, num_classes):
        super(TorchLSTMModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.lstm = nn.LSTM(vector_dim, 128, batch_first=True)  # LSTM层，输出维度128
        self.fc = nn.Linear(128, num_classes)  # 全连接层，输出字符位置类别
        self.num_classes = num_classes  # 分类数量
        self.loss = nn.CrossEntropyLoss(ignore_index=num_classes)  # 使用交叉熵损失，ignore_index用于忽略无效位置（例如没有特定字符的情况）

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        lstm_out, _ = self.lstm(x)  # (batch_size, sen_len, 128)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出 (batch_size, 128)
        output = self.fc(lstm_out)  # (batch_size, num_classes)
        if y is not None:
            return self.loss(output, y)  # 计算损失
        else:
            return output  # 只返回预测值

'''构建字符表'''
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # unk代表未知字符
    return vocab

'''判断特定字符位置'''
def build_sample(vocab, sentence_length):
    target_chars = "你我他"
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]  # 随机生成句子
    target_pos = 6 # 默认没有目标字符时设置为 -1
    for char in target_chars:
        if char in x:
            target_pos = x.index(char)  # 目标字符位置
            break
    y = target_pos  # 标签为目标字符的位置，如果没有字符，y=-1
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字符转换为索引
    return x, y

'''构建数据集'''
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

'''构建模型'''
def build_model(vocab, char_dim, sentence_length, num_classes):
    model = TorchLSTMModel(char_dim, sentence_length, vocab, num_classes)
    return model

'''测试模型准确率'''
def evaluate(model, vocab, sample_length, sentence_length, num_classes):
    model.eval()
    x, y = build_dataset(sample_length, vocab, sentence_length)  # 生成测试样本
    correct, total = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 预测
        _, predicted = torch.max(y_pred, 1)  # 取最大概率作为预测结果
        for t, p in zip(y, predicted):
            if t == p:  # 正确预测
                correct += 1
            total += 1
    accuracy = correct / total

    return accuracy

'''模型训练'''
def train():
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本数量
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本的文本长度
    num_classes = 7  # 类别数，6个位置+1（表示字符不存在的情况）
    learning_rate = 0.005  # 学习率

    # 构建词表
    vocab = build_vocab()

    # 构建模型
    model = build_model(vocab, char_dim, sentence_length, num_classes)

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        
        acc = evaluate(model, vocab, 200, sentence_length, num_classes)  # 测试模型
        log.append([acc, np.mean(watch_loss)])
        print(f"Epoch [{epoch+1}/{epoch_num}], Loss: {np.mean(watch_loss):.4f}, Acc: {acc:.4f}")

    # 绘制训练过程的准确率和损失
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 准确率
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 损失
    plt.legend()
    plt.show()

    # 保存模型和词表
    torch.save(model.state_dict(), "model.pth")
    with open("vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

'''模型预测'''
def predict(model_path, vocab_path, input_strings,sentence_length=6):
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, 20, 6, 7)  # 维度和句子长度相同
    model.load_state_dict(torch.load(model_path))
    model.eval()

    x = []
    for input_string in input_strings:
        # 确保字符串长度符合要求
        if len(input_string) < sentence_length:
            # 长度小于预设长度时，用下划线补齐
            input_string = input_string + '_' * (sentence_length - len(input_string))
        elif len(input_string) > sentence_length:
            # 长度大于预设长度时，截断字符串
            input_string = input_string[:sentence_length]

        x.append([vocab[char] for char in input_string])  # 将输入转为序列

    with torch.no_grad():
        result = model(torch.LongTensor(x))  # 模型预测
        # 对输出应用 Softmax，获得每个类别的概率
        result_prob = torch.softmax(result, dim=1)
    print("-----------------预测部分-----------------")
    for i, input_string in enumerate(input_strings):
        _, predicted = torch.max(result[i], 0)  # 取最大值作为预测结果
        predicted_prob = result_prob[i][predicted]  # 取最大概率值
        print(f"输入：{input_string}, 预测位置：{predicted.item()}, 概率：{predicted_prob.item():.4f}")

if __name__ == "__main__":
    train()
    # 位置6表示未找到特殊字符
    test_strings = ["fnvfee", "wz你你fg", "rqwdeg", "n我kwww"]
    predict("model.pth", "vocab.json", test_strings)
