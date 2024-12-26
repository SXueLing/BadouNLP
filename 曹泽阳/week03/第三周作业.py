import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt


class RNNModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab_size, num_targets):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vector_dim, padding_idx=0)
        self.rnn = nn.RNN(vector_dim, 20, batch_first=True)  # 使用 RNN，32 是隐藏层大小
        #  线性层的输入维度和rnn的输出维度必须保持一致，因为张量的计算  输出层，num_targets是类别数（包括位置+1个“不存在”类别）
        self.fc = nn.Linear(20, num_targets)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x, _ = self.rnn(x)  # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, hidden_dim)
        x = x.mean(dim=1)  # 对序列进行平均（或者可以选择最后一个时间步的输出）
        logits = self.fc(x)  # (batch_size, hidden_dim) -> (batch_size, num_targets)
        if y is not None:
            return self.loss(logits, y)
        else:
            return logits


def build_vocab():
    chars = "你我他abcdefghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length, target_chars):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    positions = [x.index(char) if char in x else -1 for char in target_chars]  # -1表示字符不在字符串中
    # 将所有位置编码为一个标签（这里简单处理为只考虑第一个出现的位置，如果有多个则取最小的索引）
    # 如果所有字符都不在，则随机选择一个不存在的位置（例如sentence_length）作为标签（这里为了简化处理）
    # 注意：这种处理方式在实际应用中可能不是最优的，因为它忽略了多个字符出现的情况。
    # 在实际应用中，可能需要更复杂的标签编码方式。
    y = min((pos for pos in positions if pos != -1), default=sentence_length)
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y


def build_dataset(sample_length, vocab, sentence_length, target_chars):
    dataset_x = []
    dataset_y = []
    num_targets = sentence_length + 1  # 包括sentence_length作为“不存在”的类别
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length, target_chars)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def evaluate(model, vocab, sample_length, target_chars):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length, target_chars)
    correct = 0
    with torch.no_grad():
        preds = model(x)
        _, predicted = torch.max(preds, 1)
        for pred, true in zip(predicted, y):
            if true == sample_length and pred == sample_length:  # 都表示“不存在”
                correct += 1
            elif true != sample_length and pred == true:  # 正确预测了位置
                correct += 1
    accuracy = correct / len(y)
    print(f"正确预测个数：{correct}, 正确率：{accuracy:.4f}")
    return accuracy


def main():
    epoch_num = 10
    batch_size = 20
    train_sample = 500
    char_dim = 20
    sentence_length = 6
    learning_rate = 0.005
    vocab = build_vocab()
    target_chars = ['你', '我', '他']
    model = RNNModel(char_dim, sentence_length, len(vocab), sentence_length + 1)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length, target_chars)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss):.4f}")
        acc = evaluate(model, vocab, sentence_length, target_chars)
        log.append([acc, np.mean(watch_loss)])

    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), "rnn_model.pth")
    json.dump(vocab, open("vocab.json", "w", encoding="utf8"), ensure_ascii=False, indent=2)


def predict(model_path, vocab_path, input_strings, target_chars):
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = RNNModel(char_dim, sentence_length, len(vocab), sentence_length + 1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()

    with torch.no_grad():
        for input_string in input_strings:
            if isinstance(input_string, str):
                 char_list = list(input_string)  # 将字符串转换为字符列表
            else:
                char_list = input_string
            x = [vocab.get(char, vocab['unk']) for char in char_list + ['pad'] * (sentence_length - len(char_list))]
            x = torch.LongTensor([x])
            logits = model(x)
            _, predicted_position = torch.max(logits, 1)
            predicted_position = predicted_position.item()
            if predicted_position == sentence_length:
                print(f"输入：{input_string}，预测结果：{', '.join(target_chars)} 不在字符串中")
            else:
                print(f"输入：{input_string}，预测结果：{', '.join(target_chars)} 在位置 {predicted_position}")


if __name__ == "__main__":
    main()
    test_strings = ["fnvfee", "wz你dfg", "rqwdeg", "n我kwww"]
    predict("rnn_model.pth", "vocab.json", test_strings, ['你', '我', '他'])
