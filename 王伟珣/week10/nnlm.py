#coding:utf8

import random
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from tqdm import tqdm


class LanguageModel(nn.Module):
    def __init__(self, bert_path):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(
            bert_path, return_dict=False, ignore_mismatched_sizes=True
        )
        hidden_size = self.bert.config.hidden_size
        vocab_size = self.bert.config.vocab_size
        
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy
        return
    
    def forward(self, x, target=None, mask=None):
        x = self.bert(x, attention_mask=mask)[0]
        y = self.classify(x)

        if target is not None:
            return self.loss(y.view(-1, y.shape[-1]), target.view(-1))
        else:
            return torch.softmax(y, dim=-1)


def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


def build_model(bert_path):
    model = LanguageModel(bert_path)
    return model


def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start: end]
    target = corpus[start + 1: end + 1]
    x = tokenizer.encode(window, max_length=window_size+2, pad_to_max_length=True)
    y = tokenizer.encode(target, max_length=window_size+2, pad_to_max_length=True)
    return x, y


def build_dataset(batch_size, tokenizer, window_size, corpus):
    dataset_x, dataset_y = [], []
    for i in range(batch_size):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def generate_sentence(openings, model, tokenizer, window_size):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        while pred_char != '\n' and len(openings) <= 30:
            openings += pred_char
            input_token = openings[-window_size:]
            x = tokenizer.encode(input_token, max_length=window_size+2, pad_to_max_length=True)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-2, :]
            index = sampling_strategy(y)
            pred_char = tokenizer.decode(index)
    return openings


def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        return int(torch.argmax(prob_distribution))
    else:
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)
    

def train():
    corpus_path = 'corpus.txt'
    bert_path = 'bert-base-chinese'
    n_epoches = 20
    batch_size = 64
    train_sample = 50000
    window_size = 10

    tokenizer = BertTokenizer.from_pretrained(bert_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Use device:", device)

    corpus = load_corpus(corpus_path)
    model = build_model(bert_path)
    if torch.cuda.is_available():
        model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(n_epoches):
        model.train()
        watch_loss = []
        for batch in tqdm(range(int(train_sample / batch_size)), total=int(train_sample / batch_size)):
            x, target = build_dataset(batch_size, tokenizer, window_size, corpus)
            batch_size, sentence_len = x.shape
            mask = torch.tril(torch.ones((batch_size, sentence_len, sentence_len), dtype=torch.bool))
            if torch.cuda.is_available():
                x, target = x.cuda(), target.cuda()
                mask = mask.cuda()
            
            optim.zero_grad()
            loss = model(x, target, mask)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, window_size))
    return


if __name__ == "__main__":
    train()
