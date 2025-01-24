
from config import Config
import torch
import torch.nn as nn

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        vocab_size = config['vocab_dict'] + 1
        hidden_size = config['hidden_size']
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, padding_idx=0)
        self.linear = nn.Linear(in_features=hidden_size,out_features=hidden_size)


    def forward(self, x_):
        x_ = self.embedding_layer(x_)   # batch_size, sentence_length, hidden_size
        x_ = self.linear(x_)            # batch_size, sentence_length, hidden_size
        x_ = torch.max_pool1d(x_.transpose(1,2), x_.shape[1]).squeeze() # batch_size, hidden_size, sentence_length -> batch_size, hidden_size, 1 -> batch_size, hidden_size
        return x_

class TripletNetwork(nn.Module):
    def __init__(self, config):
        super(TripletNetwork, self).__init__()
        self.encoder = SentenceEncoder(config)
        self.margin = config['margin']
        self.loss = self.cost_triple_loss


    def forward(self, s1, s2=None, s3=None):
        # batch_size, hidden_size
        e1 = self.encoder(s1)
        if s3 is None:
            if s2 is None:
                return self.encoder(s1)
            else:
                e2 = self.encoder(s2)
                return self.cos_dis(e1, e2)
        else:
            e2 = self.encoder(s2)
            e3 = self.encoder(s3)
            return self.loss(e1, e2, e3)

    def cos_dis(self, s1, s2):
        v_1 = nn.functional.normalize(s1, dim=-1)
        v_2 = nn.functional.normalize(s2, dim=-1)
        cos = torch.sum(torch.mul(v_1, v_2), dim=-1)
        return 1 - cos

    def cost_triple_loss(self, s1, s2, s3):
        a_v = nn.functional.normalize(s1, dim=-1)
        p_v = nn.functional.normalize(s2, dim=-1)
        n_v = nn.functional.normalize(s3, dim=-1)
        margin = self.margin
        ap = 1 - torch.sum(torch.mul(a_v, p_v), dim=-1) # batch_size, hidden_size -> batch_size,
        an = 1 - torch.sum(torch.mul(a_v, n_v), dim=-1) # batch_size, hidden_size -> batch_size,
        # return max(ap - an + margin, 0)
        diff = ap - an + margin
        return torch.mean(diff[diff.gt(0)])


def choose_optimizer(config, model):
    lr = config['learning_rate']
    if config['optimizer'] == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    else:
        return torch.optim.SGD(model.parameters(), lr=lr)


if __name__ == "__main__":
    from config import Config
    Config["vocab_dict"] = 10
    Config["max_length"] = 4
    model = TripletNetwork(Config)
    ss1 = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    ss2 = torch.LongTensor([[1,2,3,4], [3,2,3,4]])
    ss3 = torch.LongTensor([[5,6,8,1], [9,4,1,6]])
    y = model(ss1, ss2, ss3)
    y = model(ss1, ss2)
    print(y)
    # print(model.state_dict())