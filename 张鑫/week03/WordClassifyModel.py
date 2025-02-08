"""
判断特定字符在字符串的第几个位置，使用rnn和交叉熵
"""
import json
import random
import numpy
import torch
import torch.nn as nn
import matplotlib.pyplot as pl


class WordClassifyModel(nn.Module):
    def __init__(self, vocab, sentence_len, class_num):
        super(WordClassifyModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), sentence_len)
        self.rnn = nn.RNN(sentence_len, 32, bias=True, batch_first=True)
        self.layer = nn.Linear(32, class_num)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)
        _, hidden = self.rnn(x)
        y_pred = self.layer(hidden[-1])
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return torch.argmax(y_pred, dim=1)


# 构造vocab词表
def build_vocab():
    char_dict = '我你他学习'
    vocab = {'[pad]': 0}
    for i, item in enumerate(char_dict):
        vocab[item] = i + 1
    vocab['unk'] = len(char_dict)+1
    return vocab


# 转换句子为索引
def sentence_2_index(vocab, sentence):
    return [vocab.get(word, vocab['unk']) for word in sentence]


# 构建样本
def build_simple(vocab, sentence_len):
    sentence = [random.choice(list(vocab.keys())) for _ in range(sentence_len)]
    index = find_word_index("你", sentence, sentence_len)
    sentence = sentence_2_index(vocab, sentence)
    return sentence, index


def find_word_index(word, sentence, sentence_len):
    try:
        return sentence.index(word)
    except ValueError:
        return sentence_len


# 构建批量样本
def build_datasets(vocab, sentence_len, simple_size):
    x_arr = []
    y_arr = []
    for _ in range(simple_size):
        x, y = build_simple(vocab, sentence_len)
        x_arr.append(x)
        y_arr.append(y)
    return torch.LongTensor(numpy.array(x_arr)), torch.LongTensor(numpy.array(y_arr))


# 模型执行，并计算正确率
def evaluate(model, vocab, sentence_len, simple_size):
    v_x_tensor, v_y_tensor = build_datasets(vocab, sentence_len, simple_size)
    correct = 0
    with torch.no_grad():
        m_y_pred = model.forward(v_x_tensor)
        correct += (v_y_tensor == m_y_pred).sum().item()
        print(f"正确预测个数：{correct}, 正确率：{correct / simple_size :.4f}")
    return correct / simple_size


# 训练
def train(vocab_path, model_path, sentence_len, class_num):
    # 训练轮数
    epoch_num = 200
    # 每轮样本总数
    train_data_size = 500
    # 每轮中每批训练的样本数量
    train_batch_size = 50
    # 学习率
    lr = 0.001
    # 构建字典
    vocab = build_vocab()
    # 构建模型
    model = WordClassifyModel(vocab, sentence_len, class_num)
    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    log = []
    for epoch in range(epoch_num):
        x_tensor, y_tensor = build_datasets(vocab, sentence_len, train_data_size)
        model.train()
        watch_loss = []
        for i in range(train_data_size // train_batch_size):
            x = x_tensor[i * train_batch_size: (i + 1) * train_batch_size]
            y = y_tensor[i * train_batch_size: (i + 1) * train_batch_size]
            optim.zero_grad()
            loss = model.forward(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        mean_loss = numpy.mean(watch_loss)
        print(f"=======第{epoch + 1}轮训练, 平均loss为{mean_loss:.2f}=======")
        correct_rate = evaluate(model, vocab, sentence_len, train_batch_size)
        log.append([correct_rate, mean_loss])
    pl.plot(range(len(log)), [item[0] for item in log], label="acc")
    pl.plot(range(len(log)), [item[1] for item in log], label="loss")
    pl.legend()
    pl.show()
    # 保存模型
    torch.save(model.state_dict(), model_path)
    # 保存vocab
    writer = open(vocab_path, "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 验证
def predict(vocab_path, model_path, sentence_len, class_num):
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  #加载字符表
    model = WordClassifyModel(vocab, sentence_len, class_num)  #建立模型
    model.load_state_dict(torch.load(model_path, weights_only=False))  #加载训练好的权重

    test_sentence = ["我爱学你习好",
                     "你爱我学习好",
                     "我爱学习都好"]
    test_sentence_index = torch.LongTensor([sentence_2_index(vocab, i) for i in test_sentence])
    test_sentence_label = torch.LongTensor([find_word_index("你", sentence, len(sentence)) for sentence in test_sentence])
    pred_result = model.forward(test_sentence_index)
    print(f"模型预测类别: {pred_result}, 真实类别：{test_sentence_label}")


if __name__ == '__main__':
    vocab_path = "week03_vocab.json"
    model_path = "week03_model.bin"
    # 句子长度
    sentence_len = 6
    # 分类数（0到5，以及第6类：不包含'你'）
    class_num = 7

    # 训练
    train(vocab_path, model_path, sentence_len, class_num)
    # 验证
    predict(vocab_path, model_path, sentence_len, class_num)
