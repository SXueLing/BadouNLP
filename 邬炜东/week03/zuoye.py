import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json

max_length = 10
epochs = 20
batch_size = 10
vector_dim = 20
num_train = 800
num_test = 200
num_sample = 1000
learning_rate = 0.001
input_size = vector_dim
layer1_size = 10
output_size = 1
hidden_layer = 16


def build_vocab():
    chars = "我zxcvb你nm他asdfghjklqwertyuiop"
    vocab = {}
    vocab["pad"] = 0
    vocab["unk"] = 1
    for i, char in enumerate(chars):
        vocab[char] = i + 2
    return vocab


def generate_sample(vocab, max_length):
    x = [random.choice(list(vocab.keys())[2:]) for _ in range(max_length)]
    goal_str = "你我他"
    y = [0] * max_length
    for i in range(len(x)):
        if x[i] in goal_str:
            y[i] = 1
        x[i] = vocab[x[i]]
    return x, y


def build_dataset(vocab, num_sample):
    x_data = []
    y_data = []
    for i in range(num_sample):
        x, y = generate_sample(vocab, max_length)
        x_data.append(x)
        y_data.append(y)
    return torch.LongTensor(x_data), torch.FloatTensor(y_data)


class MODEL(nn.Module):
    def __init__(self, layer1_size, output_size, hidden_layer, vector_dim, vocab):
        super(MODEL, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.lstm = nn.LSTM(vector_dim, hidden_layer, num_layers=1, bidirectional=False, batch_first=True)
        self.layer1 = nn.Linear(hidden_layer, layer1_size)
        self.layer2 = nn.Linear(layer1_size, output_size)
        self.activation = torch.sigmoid
        self.loss = nn.functional.binary_cross_entropy_with_logits

    def forward(self, x, y=None):
        x = self.embedding(x)
        list_out, (hidden, cell) = self.lstm(x)  # 一般hidden_out如何往下接
        final_state = hidden[-1]
        # print(list_out.shape)
        # print("-------------------------------")
        x = self.activation(self.layer1(list_out))
        # print(x.shape)
        y_pred = self.layer2(x)
        # print(y_pred.shape)
        y_pred = y_pred.squeeze(-1)
        # 这里取巧将多标签分类任务变成了每一个时间步的0/1分类任务，类似于词性标注，按道理应该用hidden[-1]且output_size为类别个数
        # print(y_pred.shape)
        if y is not None:
            # print("-------------------------")
            # print(y_pred.shape)
            # print(y.shape)
            # print("------------------------------")
            return self.loss(y_pred, y)
        else:
            return self.activation(y_pred)


def evaluate(model, x_test, y_test):
    model.eval()
    correct = 0
    wrong = 0

    with torch.no_grad():
        y_pred = model(x_test)  # [batch_size, seq_len]

        for y_t, y_p in zip(y_test, y_pred):
            sample_correct = True

            for i in range(len(y_t)):
                if y_t[i] == 1:
                    if y_p[i] < 0.5:
                        sample_correct = False
                        break
                else:
                    if y_p[i] >= 0.5:
                        sample_correct = False
                        break

            if sample_correct:
                correct += 1
            else:
                wrong += 1

    total_samples = correct + wrong
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy


def main():
    vocab = build_vocab()
    model = MODEL(layer1_size, output_size, hidden_layer, vector_dim, vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    x_train, y_train = build_dataset(vocab, num_train)
    x_test, y_test = build_dataset(vocab, num_test)
    acc_epoch = []
    loss_epoch = []
    for epoch in range(epochs):
        model.train()
        loss_batch = []
        for index in range(num_train // batch_size):
            train_x = x_train[index * batch_size: ((index + 1) * batch_size)]
            train_y = y_train[index * batch_size: ((index + 1) * batch_size)]
            loss = model(train_x, train_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_batch.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(loss_batch)))
        acc = evaluate(model, x_test, y_test)
        acc_epoch.append(acc)
        loss_epoch.append(np.mean(loss_batch))
    plt.plot(range(len(loss_epoch)), loss_epoch, label='loss')
    plt.plot(range(len(acc_epoch)), acc_epoch, label='accuracy')
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), 'model.bin')
    writer = open("vocab.json", "w", encoding='utf-8')
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()


def predict(vocab_path, model_path, input_pred):
    vocab = json.load(open(vocab_path, "r", encoding='utf-8'))
    model = MODEL(layer1_size, output_size, hidden_layer, vector_dim, vocab)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    def preprocess_input(input_str, max_length, vocab):
        x = []
        for char in input_str[:max_length]:
            x.append(vocab.get(char, vocab["unk"]))
        x = x + [vocab["pad"]] * (max_length - len(x))
        print(x)
        return torch.LongTensor([x])

    with torch.no_grad():
        for str in input_pred:
            x = preprocess_input(str, max_length, vocab)
            y_pred = model(x)
            # print(y_pred.shape)
            y_pred = y_pred
            # print(y_pred.shape)
            print(f"输入: {str}")
            print(f"预测概率: {y_pred.tolist()}")


if __name__ == '__main__':
    main()
    # test_strings = ["fn我fee他sdf", "wz你dfghjkl", "rqwdegqwer", "你我他你我他你我他你"]
    # predict("vocab.json", "model.bin", test_strings)
