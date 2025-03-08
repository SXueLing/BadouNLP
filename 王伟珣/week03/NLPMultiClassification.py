import numpy as np
import torch
import torch.nn as nn
import random


class NLPMultiClassificationModel(nn.Module):
    def __init__(self, vocab, n_classify, vector_dim, hidden_size):
        super(NLPMultiClassificationModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.rnn = nn.RNN(vector_dim, hidden_size, bias=True, batch_first=True)
        self.linear = nn.Linear(hidden_size, n_classify)
        self.activate = nn.functional.softmax
        self.loss = nn.functional.cross_entropy
        return
    
    def forward(self, x, y=None):
        x = self.embedding(x)
        _, x = self.rnn(x)
        y_pred = self.linear(x.squeeze())
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return self.activate(y_pred, dim=1)
        return
    


def build_vocab():
    chars = "abcdefghigklmnopqrstuvwxyz"
    vocab = {"pad": 0}
    for idx, char in enumerate(chars):
        vocab[char] = idx+1
    vocab['unk'] = len(vocab)
    return vocab


def build_model(vocab, n_classify):
    model = NLPMultiClassificationModel(vocab, n_classify, 20, 8)
    return model


def str_to_squence(s, vocab):
    return [vocab.get(word, vocab['unk']) for word in s]


def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    y = x.index('a') if 'a' in x else sentence_length
    x = str_to_squence(x, vocab)
    return x, y


def build_dataset(vocab, sentence_length, n_samples):
    dataset_x, dataset_y = [], []
    for _ in range(n_samples):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def evaluate(model, vocab, sentence_length, n_sample):
    model.eval()
    x, y = build_dataset(vocab, sentence_length, n_sample)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if np.argmax(y_p) == y_t:
                correct += 1
            else:
                wrong += 1
    return correct / (correct + wrong)


def train_model(vocab, sentence_length):
    epoch_num = 100
    batch_size = 32
    train_sample = 1000
    n_classify = sentence_length+1
    learning_rate = 0.0005

    model = build_model(vocab, n_classify)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_num):
        model.train()
        epoch_loss = []
        for _ in range(int(train_sample / batch_size)):
            x, y = build_dataset(vocab, sentence_length, batch_size)
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        evaluate_acc = evaluate(model, vocab, sentence_length, 100)
        print("Epoch: %d, Avg loss: %f, Eval acc: %f" % (epoch, np.mean(epoch_loss), evaluate_acc))
    return model


def predict(model, vocab, strings):
    x = []
    for s in strings:
        x.append(str_to_squence(s, vocab))
    x = torch.LongTensor(x)

    model.eval()
    with torch.no_grad():
        result = model.forward(x)
    for i, s in enumerate(strings):
        print("Input:", s, "Output:", np.argmax(result[i]))



if __name__ == "__main__":
    print("NLP Multiple Classification ...")

    vocab = build_vocab()
    sentence_length = 6

    model = train_model(vocab, sentence_length)
    
    test_string = [
        ['a', 'b', 'd', 'e', 'f', 'a'],
        ['c', 'b', 'd', 'e', 'x', 'a'],
        ['u', 'b', 'x', 'e', 'c', 't'],
        ['h', 'b', 'a', 'o', 'k', 'p'],
    ]
    predict(model, vocab, test_string)
