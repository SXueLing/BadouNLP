import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from config import Config


class getModel(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.bert = BertModel.from_pretrained(config['bert_path'], return_dict=False)
        self.classify = nn.Linear(config['hidden_dim'], len(vocab))
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, y=None, mask=None):
        # print(mask, mask.shape)
        if y is not None:

            x, _ = self.bert(x, attention_mask = mask)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)

if __name__ == "__main__":
    from loader import load_vocab, load_corpus, build_dataset
    vocab = load_vocab(Config['vocab_path'])
    # corpus = load_corpus(Config['data_path'])
    # x, y = build_dataset(1, vocab, corpus)
    # print(x)
    model = getModel(Config, vocab)
    # causal_mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))

    # y_pred = model(x, mask=causal_mask)
    # print(y_pred.shape)
    # print(y_pred[0][0])
    # loss = model(x, y, causal_mask)
    # print(loss.item())
    pass
