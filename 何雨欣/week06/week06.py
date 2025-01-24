#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
from transformers import BertModel

# 计算Bert参数量

class TorchModel(nn.Module):
    def __init__(self, input_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        # 原始代码
        # self.embedding = nn.Embedding(len(vocab) + 1, input_dim)
        # self.layer = nn.Linear(input_dim, input_dim)
        # self.pool = nn.MaxPool1d(sentence_length)

        self.bert = BertModel.from_pretrained(r"E:\LearnSomething\NLP\week6 语言模型\bert-base-chinese", return_dict=False)

        self.classify = nn.Linear(input_dim, 3)
        self.activation = torch.sigmoid     #sigmoid做激活函数
        self.dropout = nn.Dropout(0.5)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # 原始代码
        # x = self.embedding(x)  #input shape:(batch_size, sen_len) (10,6)
        # x = self.layer(x)      #input shape:(batch_size, sen_len, input_dim) (10,6,20)
        # x = self.dropout(x)    #input shape:(batch_size, sen_len, input_dim)
        # x = self.activation(x) #input shape:(batch_size, sen_len, input_dim)
        # x = self.pool(x.transpose(1,2)).squeeze() #input shape:(batch_size, sen_len, input_dim)

        sequence_output, pooler_output = self.bert(x)

        x = self.classify(pooler_output)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y.squeeze())
        else:
            return y_pred

#字符集随便挑了一些汉字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index + 1   #每个字对应一个序号
    vocab['unk'] = len(vocab)+1
    return vocab


#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


def main():
    char_dim = 768         # 每个字的维度
    sentence_length = 6    # 样本文本长度
    vocab = build_vocab()       # 建立字表
    model = build_model(vocab, char_dim, sentence_length)    # 建立模型
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", total_params)
    return


if __name__ == "__main__":
    main()
