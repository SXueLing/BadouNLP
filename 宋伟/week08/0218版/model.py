'''
表示型文本匹配的网络搭建
1. 使用孪生网络形式进行训练

调用时：
1. 获取孪生网络，进行训练，验证，预测
2. 获取优化器

[description]
'''

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SentenceEncoder(nn.Module):
    """docstring for SentenceEncoder：
    这是一个embedding+encoder层
    目的：获取文本的张量表示
    """

    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_szie = config["hidden_size"]
        vocab_size = config["vocab_size"]+1
        max_length = config["max_length"]
        self.embedding = nn.Embedding(vocab_size, hidden_szie, padding_idx=0)
        self.layer = nn.Linear(hidden_szie, hidden_szie)
        self.max_pool = nn.MaxPool1d(max_length)
        self.dropout = nn.Dropout(0.5)  # 这个在哪里起作用，不清楚

    def forward(self, x):
        # x ： [batch,sequence]
        x = self.embedding(x)  # [batch,sequence,embedding]

        # 下面是encoder部分
        x = self.layer(x)  # [batch,sequence,embedding]
        # x = nn.functional.max_pool1d(torch.transpose(x,dim0=-1,dim1=-2),stride=x.shape[1])
        x = self.max_pool(torch.transpose(x, -1, -2))  # [batch,embedding,1]
        x = torch.squeeze(x, dim=-1)  # [batch,embedding]
        return x


class SiameseNetwork(nn.Module):
    """docstring for SiameseNetwork:
    孪生神经网络
    """

    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss_name = config["loss"]
        if config["loss"] == "triplet_loss":
            self.loss = self.cosine_triplet_loss
        elif config["loss"] == "cos_loss":
            self.loss = nn.CosineEmbeddingLoss()

    def forward(self, s1, s2=None, target=None):
        # s1 : [batch,sequence]
        # target : [batch,1]
        # 输出结果：[batch,loss],[batch,dis],[batch,embedding]
        loss = self.loss_name  # 通过全局变量进行控制，默认为所有的成员函数默认的参数，其本身可自定义起作用
        # 如果同时传入两个句子
        if loss == "cos_loss":
            if s2 is not None:
                vector_1 = self.sentence_encoder(s1)  # [batch,embedding]
                vector_2 = self.sentence_encoder(s2)
                # 如果是有标签，即训练时
                if target is not None:
                    return self.loss(vector_1, vector_2, target.squeeze())  # 计算loss,结果为标量
                # 没有标签，即预测时
                else:
                    return self.cosine_distance(vector_1, vector_2) # [batch,]
            # 如果单独传入一个句子，认为正在使用模型的向量化能力
            else:
                return self.sentence_encoder(s1)
        elif loss == "triplet_loss":
            # 进行训练时
            if not s2:
                return self.sentence_encoder(s1) # 单元素输入直接进行编码
            if target is not None:
                a = self.sentence_encoder(s1)
                p = self.sentence_encoder(s2)
                n = self.sentence_encoder(target)
                return self.loss(a,p,n)   # 标准训练，计算loss
            # 仅输入两个，可以计算它们之间的相似度（距离）
            else:
                return self.cosine_distance(a,p)  # 计算距离





    def cosine_distance(self, tensor_1, tensor_2):
        # [batch,embedding]
        tensor_1 = nn.functional.normalize(tensor_1, dim=-1)
        tensor_2 = nn.functional.normalize(tensor_2, dim=-1)
        cosine = torch.sum(torch.mul(tensor_1, tensor_2), dim=-1)  # [batch,1]
        return 1 - cosine  # 这里使用了广播机制

    def cosine_triplet_loss(self, a, p, n, margin=None):
        # a ：[batch,embedding]
        # p:[batch,embedding]
        # n :[batch,embedding]
        ap = self.cosine_distance(a, p)  # [batch,1]
        an = self.cosine_distance(a, n) # [batch,1]
        if margin is None:
            diff = ap- an + 0.1   # [batch,1]
        else:
            diff = ap - an + margin.sequence() # [batch,1]
        out = diff[diff.gt(0)] # [baych,1],dtype = bool,[n]
        return torch.mean(out)  # 这里的gt 是greater than,即返回批量数据的平均loss,即标量，=


def choose_optimizer(config,model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(),lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(),lr=learning_rate)

if __name__ == '__main__':
    from config import Config
    # 这里的测试不会改写原配置数据
    Config["vocab_size"] = 10
    Config["max_length"] = 4
    Config["loss"] = "triplet_loss"
    model = SiameseNetwork(Config)
    s1 = torch.randint(0,Config["vocab_size"],[12,4])
    s2 = torch.randint(0,Config["vocab_size"],[12,4])
    # label = torch.tensor([0,1])[torch.randint(0,2,[12,1])]
    s3 = torch.randint(0,Config["vocab_size"],[12,4])
    # out = model(s1,s2,label)  # 这个batch_size 的平均loss
    out = model(s1,s2,s3)  # 这个batch_size 的平均loss
    print(out)