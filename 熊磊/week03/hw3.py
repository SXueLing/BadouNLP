import torch
import torch.nn as nn
import random
import numpy as np

# 找一个特定字符在哪

class NLPMultiClass(nn.Module):
    def __init__(self, vocab, embedding_size, hidden_size, num_layers, n_class):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), embedding_size, padding_idx = 0)
        self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, batch_first = True)
        self.lienar = nn.Linear(hidden_size, n_class)
        self.loss = nn.functional.cross_entropy
    
    def forward(self, x, y = None):
        x = self.embedding(x)
        _, x = self.rnn(x)
        y_pred = self.lienar(x.squeeze())
        if y is not None:

            return self.loss(y_pred, y)
        else:
            return y_pred

def buildVocab():
    char = 'abcdefghijklmnopqrstuvwxyz'
    vocab = {'': 0}
    for i in range(len(char)):
        vocab[char[i]] = i + 1
    vocab['UNK'] = len(char) + 1
    return vocab

def buildSample(vocab, sentence_length):
    # x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    x = random.sample(list(vocab.keys()), sentence_length)
    if 'a' in x:
        y = x.index('a')
    else:
        y = sentence_length
    x = [vocab.get(word, vocab['UNK']) for word in x]
    return x, y

def buildDataset(vocab, sentence_length, n):
    dataset_x = []
    dataset_y = []
    for _ in range(n):
        x, y = buildSample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)
    
def evaluate(model, vocab, sentence_length, n):
    model.eval()
    x, y_true = buildDataset(vocab, sentence_length, n)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_t, y_p in zip(y_true, y_pred):
            if y_t == torch.argmax(y_p):
                correct += 1
            else:
                wrong += 1
    return correct / (correct + wrong)
        

    
def main():
    epoch_num = 10
    batch_size = 32
    learning_rate = 1e-3
    sentence_length = 6
    n = 512
    vocab = buildVocab()
    embedding_size = 16
    hidden_size = 32
    num_layers = 1

    model = NLPMultiClass(vocab, embedding_size, hidden_size, num_layers, sentence_length + 1)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for _ in range(int(n / batch_size)):
            x, y = buildDataset(vocab, sentence_length, batch_size)

            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print('------------------')
        acc = evaluate(model, vocab, sentence_length, 100)
        print(f"第{epoch + 1}轮的平均损失为{np.mean(watch_loss)}，正确率为{acc}")
    return model

def predict(model, vocab, samples):
    x = []
    for sample in samples:
        temp = [vocab.get(word, vocab['UNK']) for word in sample]
        x.append(temp)
    model.eval()
    with torch.no_grad():
        result = model(torch.LongTensor(x))
    for i, input_string in enumerate(samples):
        print(f"输入：{input_string}, 预测类别：{torch.argmax(result[i])}")


if __name__ == "__main__":
    model = main()
    vocab = buildVocab()
    
    test_string = [
        ['a', 'b', 'd', 'e', 'f', '我'],
        ['c', 'b', 'd', 'e', 'x', 'a'],
        ['u', 'b', 'x', 'e', 'c', 't'],
        ['h', 'b', 'a', 'o', 'k', 'p'],
    ]
    predict(model, vocab, test_string)
