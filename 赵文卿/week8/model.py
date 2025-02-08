'''
Author: Zhao
Date: 2025-01-14 19:55:10
LastEditTime: 2025-01-15 18:24:32
FilePath: model.py
Description: 实现Triplet Loss
        SentenceEncoder: 将输入句子编码为固定长度的向量。
        
        SiameseNetwork: 使用三元组损失计算句子向量之间的差异。
        
        choose_optimizer: 根据配置选择优化器(Adam 或 SGD)。

'''
import torch
import torch.nn as nn
from torch.optim import Adam, SGD

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        """
        初始化模型，根据配置选择不同的编码器
        :param config: 配置字典，包含各种超参数和路径
        """
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        # 初始化嵌入层，词汇表大小为 vocab_size，嵌入维度为 hidden_size，0 作为填充索引
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, padding_idx=0)
        # 初始化线性层，将输入的 hidden_size 映射到 hidden_size
        self.layer = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        # 初始化 Dropout 层，防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 将输入 x 映射到嵌入空间
        x = self.embedding(x)
        # 使用线性层进行映射
        x = self.layer(x)
        # 对 x 进行 max pooling，压缩序列长度
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        # 初始化句子编码器
        self.sentence_encoder = SentenceEncoder(config)
        # 初始化三元组损失
        self.loss = nn.TripletMarginLoss(margin=1.0, p=2)  # 可以根据需要调整margin和p参数

    # 计算三元组损失
    def triplet_loss(self, anchor, positive, negative):
        return self.loss(anchor, positive, negative)
    
    def forward(self, anchor, positive=None, negative=None):
        anchor_vector = self.sentence_encoder(anchor)
        if positive is not None and negative is not None:
            positive_vector = self.sentence_encoder(positive)
            negative_vector = self.sentence_encoder(negative)
            return self.triplet_loss(anchor_vector, positive_vector, negative_vector)
        else:
            return anchor_vector

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        # 返回 Adam 优化器
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        # 返回 SGD 优化器
        return SGD(model.parameters(), lr=learning_rate)