import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, vector_dim, vocab, num_classes, dropout_prob=0.5):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.rnn = nn.RNN(input_size=vector_dim, hidden_size=32, bias=False, batch_first=True)  # RNN层
        self.dropout = nn.Dropout(p=dropout_prob)  # 应用 dropout
        self.classify = nn.Linear(32, num_classes)  # 线性层
        self.softmax = nn.Softmax(dim=1)  # softmax归一化函数
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x, _ = self.rnn(x)     # (batch_size, sen_len, 32) 注意这里的第三个维度现在是32了
        x = x[:, -1, :]        # 取最后一个时间步的输出作为句子表示
        x = self.dropout(x)    # 应用 dropout
        x = self.classify(x)   # (batch_size, num_classes)
        y_pred = self.softmax(x)  # (batch_size, num_classes)
        if y is not None:
            return self.loss(y_pred, y.squeeze())  # 计算损失
        else:
            return y_pred  # 输出预测结果的概率分布


def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    target_chars = "你我他"
    include_target = random.choice([True, False])

    if include_target:
        x = [random.choice(list(vocab.keys())) for _ in range(sentence_length - 1)]
        target_char = random.choice(target_chars)
        position = random.randint(0, sentence_length - 1)
        x.insert(position, target_char)
        label = position
    else:
        non_targets = [char for char in vocab if char not in target_chars]
        x = [random.choice(non_targets) for _ in range(sentence_length)]
        label = sentence_length

    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, label


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_model(vocab, char_dim, num_classes):
    model = TorchModel(char_dim, vocab, num_classes)
    return model


def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)
    correct, total = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        _, predicted = torch.max(y_pred, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)
    accuracy = correct / total
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy


def main():
    epoch_num = 40
    batch_size = 20
    train_sample = 500
    char_dim = 20
    sentence_length = 6
    learning_rate = 0.005
    num_classes = sentence_length + 1  # 每个位置作为一个类别，包括不存在的情况

    vocab = build_vocab()
    model = build_model(vocab, char_dim, num_classes)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)

            # 打印每个批次的样本和标签
            print(f"Epoch {epoch + 1}, Batch {batch + 1} samples and labels:")
            for idx in range(len(x)):
                print(f"Sample {idx + 1}: {x[idx].tolist()}, Label: {y[idx].item()}")

            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])

    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), "model.pth")
    writer = open("vocab.json", "w", encoding="utf8")
    json.dump(vocab, writer, ensure_ascii=False, indent=2)
    writer.close()


def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    num_classes = sentence_length + 1  # 每个位置作为一个类别，包括不存在的情况
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()

    # 将输入字符串填充到固定长度，并转换为索引序列
    x = [[vocab.get(char, vocab['unk']) for char in input_string.ljust(sentence_length)] for input_string in input_strings]

    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
        probabilities = result.numpy()

        for i, input_string in enumerate(input_strings):
            prob_dist = probabilities[i]
            predicted_class = np.argmax(prob_dist)  # 获取最大概率对应的索引

            if predicted_class == num_classes - 1:
                position = "不存在"
            else:
                position = str(predicted_class)

            print("输入：%s, 预测位置：%s, 预测概率分布：%s" % (input_string, position, prob_dist))


if __name__ == "__main__":
    main()
    test_strings = ["fnvfee", "wz你dfg", "rqwdeg", "n我kwww", "nkwww他", "nkssw谁", "dgasd哭"]
    predict("model.pth", "vocab.json", test_strings)
