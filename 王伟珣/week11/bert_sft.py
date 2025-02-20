#coding:utf8

import random
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from tqdm import tqdm


class LanguageModel(nn.Module):
    def __init__(self, bert_path):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(
            bert_path, return_dict=False, attn_implementation='eager'
        )
        hidden_size = self.bert.config.hidden_size
        vocab_size = self.bert.config.vocab_size
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy
        return
    
    def forward(self, x, target=None, mask=None):
        if target is not None:
            x, _ = self.bert(x, attention_mask=mask)
            y = self.classify(x)
            return self.loss(y.view(-1, y.shape[-1]), target.view(-1))
        else:
            x, _ = self.bert(x)
            y = self.classify(x)
            return torch.softmax(y, dim=-1)
        

class DataGenerator(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = []
        with open(data_path, encoding='utf8') as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                title, content = line['title'], line['content']
                self.prepare_data(title, content)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def prepare_data(self, title, content):
        x = title + '[SEP]' + content
        y = title[1:] + '[SEP]' + content + 'eos'
        x = self.tokenizer.encode(
            x, add_special_tokens=False, padding='max_length', truncation=True, max_length=self.max_seq_len
        )
        y = self.tokenizer.encode(
            y, add_special_tokens=False, padding='max_length', truncation=True, max_length=self.max_seq_len
        )
        y[:len(title)] = [-1e2] * len(title)

        mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len))
        mask[:len(title)+1, :len(title)+1] = 1.0

        self.data.append(
            [torch.LongTensor(x), torch.LongTensor(y), mask]
        )
        return
    

def load_data(data_path, tokenizer, max_seq_len, batch_size, shuffle=True):
    dataset = DataGenerator(data_path, tokenizer, max_seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def generate_sentence(openings, model, tokenizer, max_seq_len):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        while pred_char != 'eos' and len(openings) <= max_seq_len:
            openings += pred_char
            x = tokenizer.encode(openings, add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = ''.join(tokenizer.decode(index))
    return openings


def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        return int(torch.argmax(prob_distribution))
    else:
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


def train():
    sample_path = 'sample_data.json'
    bert_path = 'bert-base-chinese'
    max_seq_len = 128
    n_epoches = 20
    batch_size = 16
    lr = 1e-3
    
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    train_data = load_data(sample_path, tokenizer, max_seq_len, batch_size, True)

    model = LanguageModel(bert_path)
    if torch.cuda.is_available():
        model = model.cuda()
    
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epoches):
        model.train()
        train_loss = []
        for _, (x, target, mask) in tqdm(enumerate(train_data), total=len(train_data)):
            if torch.cuda.is_available():
                x, target, mask = x.cuda(), target.cuda(), mask.cuda()
            optim.zero_grad()
            loss = model(x, target, mask)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(train_loss)))
        print(generate_sentence("各路足坛名人陆续抵达", model, tokenizer, max_seq_len))
        print(generate_sentence("用别人的卡取钱 是提醒还是偷盗？", model, tokenizer, max_seq_len))
    return


if __name__ == '__main__':
    train()
