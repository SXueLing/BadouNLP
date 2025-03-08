import torch
import torch.nn as nn
from torch.optim import Adam, SGD


class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config['hidden_size']
        vocab_size = config['vocab_size']
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x = self.layer(x)
        input = x.transpose(1, 2)
        ke = x.shape[1]
        x = nn.functional.max_pool1d(input, ke).squeeze()
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.encoder = SentenceEncoder(config)
        self.loss = nn.CosineEmbeddingLoss()

    # def forward(self, sentence1, sentence2=None, target=None):
    #     # 同时传入两个句子
    #     if sentence2 is not None:
    #         vector1 = self.encoder(sentence1)  # vec:(batch_size, hidden_size)
    #         vector2 = self.encoder(sentence2)
    #         # 如果有标签，则计算loss
    #         if target is not None:
    #             return self.loss(vector1, vector2, target.squeeze())
    #         # 如果无标签，计算余弦距离
    #         else:
    #             return self.cosine_distance(vector1, vector2)
    #     # 单独传入一个句子时，认为正在使用向量化能力
    #     else:
    #         return self.encoder(sentence1)

    def forward(self, sentence1, sentence2=None, sentence3=None):
        #同时传入3个句子,则做tripletloss的loss计算
        if sentence2 is not None and sentence3 is not None:
            vector1 = self.encoder(sentence1)
            vector2 = self.encoder(sentence2)
            vector3 = self.encoder(sentence3)
            return self.cosine_triplet_loss(vector1, vector2, vector3)
        #单独传入一个句子时，认为正在使用向量化能力
        else:
            return self.encoder(sentence1)


    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    def cosine_triplet_loss(self, a, p, n, margin=None):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin.squeeze()
        return torch.mean(diff[diff.gt(0)])  # greater than

def choose_optimizer(model, config):
    rate = config['learning_rate']
    if config['optimizer'] == 'SGD':
        return SGD(model.parameters(), lr=rate)
    elif config['optimizer'] == 'adam':
        return Adam(model.parameters(), lr=rate)

if __name__ == "__main__":
    from config import Config
    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = SiameseNetwork(Config)
    s1 = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    s2 = torch.LongTensor([[1,2,3,4], [3,2,3,4]])
    l = torch.LongTensor([[1],[0]])
    y = model(s1, s2)
    print(y)

