import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel
import json
import math


class BertNet(nn.Module):
    def __init__(self, config_path):
        super(BertNet, self).__init__()
        self.config = self.load_config(config_path)
        self.embeding = BertEmbeding(self.config)
        self.encoder = BertEncoder(self.config)
        return

    
    def forward(self, x):
        embeding = self.embeding(x)
        encoder = self.encoder(embeding)
        return encoder
    
    def load_config(self, config_path):
        with open(config_path) as f:
            return json.load(f)
    

class BertEmbeding(nn.Module):
    def __init__(self, config):
        super(BertEmbeding, self).__init__()
        self.vocab_size = config['vocab_size']
        self.type_vocab_size = config['type_vocab_size']
        self.max_position_embeddings = config['max_position_embeddings']
        self.hidden_size = config['hidden_size']
        self.layer_norm_eps = config['layer_norm_eps']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.pad_token_id = config['pad_token_id']
    
        self.word_embeding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=self.pad_token_id)
        self.token_embeding = nn.Embedding(self.type_vocab_size, self.hidden_size)
        self.position_embeding = nn.Embedding(self.max_position_embeddings, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps, elementwise_affine=True)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, x):
        x = x[:self.max_position_embeddings] if len(x) >= self.max_position_embeddings else x
        position_ids = np.array([self.max_position_embeddings - 1] * self.max_position_embeddings)
        position_ids[:len(x)] = np.array(list(range(len(x))))
        we = self.word_embeding(x)
        pe = self.position_embeding(torch.LongTensor(position_ids))[: len(x)]
        te = self.token_embeding(torch.LongTensor(np.array([0] * len(x))))
        embeding = we + pe + te
        embeding = self.layer_norm(embeding)
        embeding = self.dropout(embeding)
        return embeding
    

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.n_layers = config['num_hidden_layers']
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(self.n_layers)])
        return

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.hidden_size = config['hidden_size']
        self.layer_norm_eps = config['layer_norm_eps']

        self.self_attention = BertSelfAttention(config)
        self.layer_norm_0 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps, elementwise_affine=True)
        self.feed_forward = BertFeedForward(config)
        self.layer_norm_1 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps, elementwise_affine=True)
        return

    def forward(self, x):
        o1 = self.self_attention(x)
        o1 = o1 + x
        o1 = self.layer_norm_0(o1)
        o2 = self.feed_forward(o1)
        o2 = o2 + o1
        out = self.layer_norm_1(o2)
        return out


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        self.num_attention_heads = config['num_attention_heads']
        self.hidden_size = config['hidden_size']
        self.attention_probs_dropout_prob = config['attention_probs_dropout_prob']
        self.layer_norm_eps = config['layer_norm_eps']

        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.dk = math.sqrt(self.attention_head_size)

        self.Q = nn.Linear(self.hidden_size, self.hidden_size)
        self.K = nn.Linear(self.hidden_size, self.hidden_size)
        self.V = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout_0 = nn.Dropout(self.attention_probs_dropout_prob)
        self.softmax = nn.Softmax()

        self.linear = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.dropout_1 = nn.Dropout(self.attention_probs_dropout_prob)
        return
    
    def forward(self, x):
        q = self.transpose_for_scores(self.Q(x))
        k = self.transpose_for_scores(self.K(x))
        v = self.transpose_for_scores(self.V(x))
        qk = torch.matmul(q, k.swapaxes(1, 2))
        qk /= self.dk
        qk = self.softmax(qk)
        qkv = torch.matmul(qk ,v)
        qkv = qkv.swapaxes(0, 1).reshape(-1, self.hidden_size)
        o1 = self.dropout_0(qkv)
        o2 = self.linear(o1)
        out = self.dropout_1(o2)
        return out


    def transpose_for_scores(self, x):
        max_len, _ = x.shape
        x = x.reshape(max_len, self.num_attention_heads, self.attention_head_size)
        x = x.swapaxes(1, 0)
        return x
    

class BertFeedForward(nn.Module):
    def __init__(self, config):
        super(BertFeedForward, self).__init__()
        self.hidden_size = config['hidden_size']
        self.intermediate_size = config['intermediate_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.linear_0 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.gelu = nn.GELU()
        self.linear_1 = nn.Linear(self.intermediate_size, self.hidden_size)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
    
    def forward(self, x):
        x = self.linear_0(x)
        x = self.linear_1(x)
        x = self.gelu(x)
        out = self.dropout(x)
        return out



if __name__ == "__main__":
    x = torch.LongTensor(np.array([2450, 15486, 102, 2110]))

    # bert_pretrained = BertModel.from_pretrained(
    #     r'C:\Learning\BADOU\NLP课程\BADOU_NLP_SOURCES\datas\BERT\bert-base-chinese', return_dict=False
    # )

    bert = BertNet(
        config_path=r'C:\Learning\BADOU\NLP课程\BADOU_NLP_SOURCES\datas\BERT\bert-base-chinese\config.json'
    )

    # print(bert)
    bert.eval()
    out = bert(x)
    print(out)

    # print(len(
    #     torch.nn.utils.parameters_to_vector(
    #         [p for p in bert.parameters() if p.requires_grad]
    #     )
    # ))

    # for name, param in bert.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data.size(), torch.prod(torch.tensor(param.data.size())))
