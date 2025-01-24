# coding:utf8
# auther: 王良顺

import random

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#随机生成一个样本
#从所有字中选取sentence_length个字
#y值为'你'字的下标,若未出现则记为unk
def build_sample(vocab, sentence_length):
    #指定哪些字出现时为正样本
    key = "你"
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if key in x :
        y = x.index(key)
    else:
        y = sentence_length
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x).to(device), torch.LongTensor(dataset_y).to(device)
