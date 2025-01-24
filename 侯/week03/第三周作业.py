"""
多分类任务：
'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', '你', '我', '他'
’你‘字符是第几位，就是第几类， 不存在就是第0类，依次类推
"""

import json
import random

import torch
import torch.nn as nn
import numpy as np


def build_vocab():
    target_str = '你我他'
    letters = [chr(i) for i in range(ord('a'), ord('e') + 1)]
    letters.append(target_str)
    latin_list = ''.join(letters)
    vocab_ = {'[pad]': 0}
    vocab_choice_ = {}
    for i, s in enumerate(latin_list):
        vocab_[s] = i + 1
        vocab_choice_[s] = i + 1
    vocab_['[unk]'] = len(latin_list) + 1
    return vocab_, vocab_choice_


def vocab_to_index(vocab_, seq_):
    return [vocab_.get(s, vocab_['[unk]']) for s in seq_]


def build_sample(vocab_, latin_size_, target_latin_):
    # 得到 'wdkc你shd'
    x = [random.choice(list(vocab_.keys())) for i in range(latin_size_)]
    # print('x:\n', x)
    # 例:wdkc你shd
    if x.count(target_latin_) == 0:
        y = 0
    else:
        y = x.index(target_latin_) + 1


    x = [vocab_.get(s) for s in x]
    # print('x:\n', x)
    # print('y:\n', y)
    return x, y


def build_dataset(sample_size_, vocab_, latin_size_, target_latin_):
    X_ = []
    Y_ = []
    for i in range(sample_size_):
        x, y = build_sample(vocab_, latin_size_, target_latin_)
        X_.append(x)
        Y_.append(y)
    # print(X_)
    # print(Y_)
    return X_, Y_


class MyTorchModule(nn.Module):
    def __init__(self, s_vocab_num, s_latin_dim, RNN_input_size, RNN_hidden_size):
        super(MyTorchModule, self).__init__()
        self.embedding_linear = nn.Embedding(s_vocab_num, s_latin_dim, padding_idx=0)
        self.linear_RNN = nn.RNN(input_size=RNN_input_size, hidden_size=RNN_hidden_size, bias=False, batch_first=True)
        self.loss = nn.functional.cross_entropy

    def forward(self, X_, Y_=None):
        X_embedding = self.embedding_linear(X_)     # x_:[20, 5] -> [20, 5, 16]
        Y_pred, y_pred_las = self.linear_RNN(X_embedding)    # [20, 5, 16] -> []
        # y_pred_las_sque = y_pred_las.squeeze()
        y_pred_las_sque = Y_pred[:, -1, :]      # [:, -1, :] 得到所有batch的，倒数第一行的，所有列。
                                                # 意义是最后一个结果包含整句的信息，所以从5变为1，最终合并，所以减少了一个维度 [20, 16]
        if Y_ is None:
            return y_pred_las_sque
        else:
            return self.loss(y_pred_las_sque, Y_)


def build_module(s_vocab_num, s_latin_dim, RNN_input_size, RNN_hidden_size):
    return MyTorchModule(s_vocab_num, s_latin_dim, RNN_input_size, RNN_hidden_size)

def eva(myModule, vocab_, latin_size_, target_latin_):
    sample_num = 100
    eva_x, eva_y = build_dataset(sample_num, vocab_, latin_size_, target_latin_)
    correct = 0
    wrong = 0
    with torch.no_grad():
        myModule.eval()
        eva_y_pred = myModule.forward(torch.LongTensor(eva_x))
        for i, j in zip(eva_y_pred, eva_y):
            i1 = list(i)
            i2 = i1.index(max(i1))
            if i2 == j:
                correct += 1
            else:
                wrong += 1
            # print(f'y_p：{i}, i2:{i2}, y_t：{j}')

    print(f'正确预测：{correct}个, 错误预测：{wrong}个')

def predict(module_path, pre_input, vocab_path):
    vocab = json.load(open(vocab_path, 'r', encoding='utf8'))
    latin_dim = 16
    seq_list = 5
    RNN_input_size = 16
    RNN_hidden_size = 16
    preModule = build_module(len(vocab), latin_dim, RNN_input_size, RNN_hidden_size)
    preModule.load_state_dict(torch.load(module_path))
    pre_X = []
    for e in pre_input:
        pre_X.append([vocab.get(char, vocab['[unk]']) for char in e])
    preModule.eval()
    with torch.no_grad():
        pre_Y = preModule(torch.LongTensor(pre_X))
    for i, res_Y in zip(pre_input, pre_Y):
        res = res_Y.tolist()
        print("输入:{},预测:{}".format(i, res.index(res_Y.max())))


def main():
    vocab, vocab_choice = build_vocab()
    print(vocab)
    latin_dim = 16
    seq_len = 5
    sample_size = 2000
    epoch_num = 30
    batch_size = 20
    target_latin = '你'
    RNN_input_size = 16
    RNN_hidden_size = 16
    lr = 0.001
    train_X, train_Y_true = build_dataset(sample_size, vocab, seq_len, target_latin)
    print('X:\n', train_X)
    # print('X:\n', torch.LongTensor(train_X).shape)
    print('Y:\n', train_Y_true)
    # print('Y:\n', torch.LongTensor(train_Y_true).shape)
    print('len(vocab):\n', len(vocab))
    module = MyTorchModule(len(vocab), latin_dim, RNN_input_size, RNN_hidden_size)
    optim = torch.optim.Adam(module.parameters(), lr=lr)

    for epoch in range(epoch_num):
        module.train()
        epoch_loss = []
        for batch_index in range(sample_size // batch_size):
            train_batch_X = train_X[batch_index * batch_size: (batch_index + 1) * batch_size]
            train_batch_Y = train_Y_true[batch_index * batch_size: (batch_index + 1) * batch_size]

            loss = module(torch.LongTensor(train_batch_X), torch.LongTensor(train_batch_Y))
            loss.backward()
            optim.step()
            optim.zero_grad()
            epoch_loss.append(loss.item())
        eva(module, vocab, seq_len, target_latin)
        print(f'第{epoch}轮的损失数：{np.mean(epoch_loss)}')

    torch.save(module.state_dict(), 'week03_homework_DAY02_06.bin')
    writer = open("week03_homework_vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    # writer = open("week03_homework_vocab_choice.json", "w", encoding="utf8")
    # writer.write(json.dumps(vocab_choice, ensure_ascii=False, indent=2))
    # writer.close()
    return


if __name__ == "__main__":
    # vocab, vocab_choice = build_vocab()
    # # build_sample(vocab_choice, 8, '你')
    # build_dataset(10, vocab_choice, 8, '你')
    # main()
    xi = ['wegnr', 'nfpiq', 'cnshd', 'c你shd', 'aeshd', 'wd你cd', 'cash你']
    predict('week03_homework_DAY02_06.bin', xi, 'week03_homework_vocab.json')


