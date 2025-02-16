#使用Bert完成自回归语言模型训练。不用LSTM
import os.path
import torch
import torch.nn as nn
import numpy as np
import math
import random
import re
from transformers import BertModel, BertConfig

class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained('D:\\桌面\\资料\\week6 语言模型和预训练\\bert-base-chinese')
        self.classify = nn.Linear(input_dim, len(vocab))
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None, attention_mask=None):
        outputs = self.bert(x, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        y_pred = self.classify(sequence_output)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)

def build_vocab(vocab_path):
    vocab = {'<pad>': 0}
    with open(vocab_path, encoding='utf8') as f:
        for index, line in enumerate(f):
            char = line[:-1]
            vocab[char] = index + 1
    return vocab

def load_corpus(path):
    corpus = ''
    with open(path, encoding='gbk') as f:
        for line in f:
            corpus += line.strip()
    return corpus

def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]
    x = [vocab.get(word, vocab['<UNK>']) for word in window]
    y = [vocab.get(word, vocab['<UNK>']) for word in target]
    attention_mask = torch.tril(torch.ones(window_size, window_size))  # 下三角矩阵
    return x, y, attention_mask

def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    attention_masks = []
    for i in range(sample_length):
        x, y, attention_mask = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
        attention_masks.append(attention_mask)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y), torch.stack(attention_masks)

def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model

def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ''
        while pred_char != '\n' and len(openings) <= 100:
            openings += pred_char
            x = [vocab.get(char, vocab['<UNK>']) for char in openings[-window_size:]]
            seq_len = len(x)
            attention_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0) #可变长
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
                attention_mask = attention_mask.cuda()
            y = model(x, attention_mask=attention_mask)[0][-1]
            index = sampling_strategy(y)
            pred_char = reverse_vocab[index]
    return openings

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = 'greedy'
    else:
        strategy = 'sampling'

    if strategy == 'greedy':
        return int(torch.argmax(prob_distribution))
    elif strategy == 'sampling':
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)

def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab['<UNK>']) for char in window]
            seq_len = len(x)
            # 生成下三角注意力掩码
            attention_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab['<UNK>'])
            if torch.cuda.is_available():
                x = x.cuda()
                attention_mask = attention_mask.cuda()
            pred_prob_distribute = model(x, attention_mask=attention_mask)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))

def train(corpus_path, save_weight=True):
    epoch_num = 10
    batch_size = 64
    train_sample = 50000
    char_dim = 768
    window_size = 10
    vocab = build_vocab('vocab.txt') #建立词表映射
    corpus = load_corpus(corpus_path)   #加载语料
    model = build_model(vocab, char_dim)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    print('文本词表模型加载完毕，开始训练')
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y, attention_mask = build_dataset(batch_size, vocab, window_size, corpus)
            if torch.cuda.is_available():
                x, y, attention_mask = x.cuda(), y.cuda(), attention_mask.cuda()
            optim.zero_grad()
            loss = model(x, attention_mask=attention_mask, y=y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print('========\n第%d轮平均loss:%f' % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence('让他在半年之前，就不能做出', model, vocab, window_size))
        print(generate_sentence('李慕站在山路上，深深的呼吸', model, vocab, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace('txt', 'pth')
        model_path = os.path.join('model', base_name)
        torch.save(model.state_dict(), model_path)
        return

if __name__ == '__main__':
    train('corpus.txt', False)
