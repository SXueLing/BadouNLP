import torch
import torch.nn as nn
import torch.optim as optim
import random
import json

"""
1.得到数据集
2.定义模型
3.训练模型
4.测试模型
"""

vocab_path = './vocab.json'
model_path = './test.pth'


# 神经网络
class Net(nn.Module):
    def __init__(self, vocab_size, word_dim, hidden_size, words_len):
        super(Net, self).__init__()
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.hidden_size = hidden_size
        self.words_len = words_len
        self.word_embed = nn.Embedding(vocab_size, word_dim)
        self.rnn = nn.RNN(word_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, words_len)

    def forward(self, x, y=None):
        x = self.word_embed(x)
        _, state = self.rnn(x)
        output = self.fc(state)
        return output


def build_dataset(batchs, batch_size, words_len):
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    for i in range(batchs):
        X = []
        Y = []
        for j in range(batch_size):
            x = [random.randint(0, len(vocab) - 2) for _ in range(words_len)]
            y = random.randint(0, words_len - 1)
            x[y] = vocab['你']  # 确保vocab字典中有'你'这个键
            X.append(x)
            Y.append(y)
        yield torch.tensor(X), torch.tensor(Y)


# 测试代码
def evaluate(net, batchs, batch_size, words_len, cross_entropy):
    net.eval()  # 测试模式
    val_loss, val_acc = 0, 0,
    with torch.no_grad():
        for x, y in build_dataset(batchs, batch_size, words_len):
            out = net(x).squeeze()
            loss = cross_entropy(out, y.long())
            val_loss += loss.item()
            val_acc += (out.argmax(-1) == y).sum().item()
        return val_loss, val_acc


def train(vocab, epochs, word_dim, hidden_size, words_len):
    batch_size = 32
    batchs = 100

    # 创建网络
    net = Net(len(vocab), word_dim, hidden_size, words_len)
    # 创建优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    cross_entropy = nn.CrossEntropyLoss()

    #     开始训练
    for epoch in range(epochs):
        loss_sum = 0
        acc_sum = 0
        for x, y in build_dataset(batchs, batch_size, words_len):
            optimizer.zero_grad()
            out = net(x)

            loss = cross_entropy(out.squeeze(), y.long())
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            # 使用正确的方法计算准确率
            acc_sum += (out.argmax(-1) == y).sum().item()

        #     验证数据集
        val_loss, val_acc = evaluate(net, batchs, batch_size, words_len, cross_entropy)
        # print("epoch:{},    loss:{},    acc:{}".format(epoch + 1, loss_sum / batchs, acc_sum / (batchs * batch_size)))

        print(f"Epoch: {epoch + 1}, Train Loss: {loss_sum / batchs}, Train Acc: {acc_sum / (batchs * batch_size)}, "
              f"Val Loss: {val_loss / batchs}, Val Acc: {val_acc / (batchs * batch_size)}")

        # 保存模型
    torch.save(net.state_dict(), model_path)


def test(vocab, word_dim, hidden_size, words_len):
    # 创建网络
    net = Net(len(vocab), word_dim, hidden_size, words_len)
    net.load_state_dict(torch.load(model_path, weights_only=True))

    net.eval()  # 测试模式
    with torch.no_grad():
        for x, y in build_dataset(1, 4, words_len):
            out = net(x).squeeze()
            pred = out.argmax(-1)

            # 反转vocab字典以从索引获取字符
            reversed_vocab = {v: k for k, v in vocab.items()}

            # 将张量转换为字符列表
            x_char_lists = [[reversed_vocab[j.item()] for j in i] for i in x]

            # 打印字符列表
            for char_list in x_char_lists:
                print(char_list)

            print('预测结果：', pred)
            print('真实结果:', y)


if __name__ == '__main__':
    words_len = 6
    word_dim = 100
    hidden_size = 256
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    train(vocab, 10, word_dim, hidden_size, words_len)
    # test(vocab, word_dim, hidden_size, words_len)
