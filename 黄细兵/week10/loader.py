import random

import torch

class DataLoader():
    def __init__(self, config, vocab):
        config['vocab_size'] = len(vocab)

def load_data(config, corpus, tokenizer):
    vocab_path = config['vocab_path']
    corpus_path = config['corpus_path']
    batch_size = config['batch_size']
    window_size = config['window_size']
    # 加载词表
    vocab = load_vocab(vocab_path)
    # 加载语料
    # corpus = load_corpus(corpus_path)
    config['vocab_size'] = len(vocab)
    return load(batch_size, window_size, corpus, tokenizer)

# 随机获取训练数据
def load(batch_size, window_size, corpus, tokenizer):
    x = []
    y = []
    for i in range(batch_size):
        start = random.randint(0, len(corpus) - 1 - window_size)
        end = start + window_size
        startText = corpus[start:end]
        endText = corpus[start + 1: end + 1]
        # x_data = [vocab.get(char, vocab["<UNK>"]) for char in startText]
        # y_data = [vocab.get(char, vocab["<UNK>"]) for char in endText]
        x_data = tokenizer.encode(startText, add_special_tokens=False, padding='max_length', truncation=True,max_length=10)  # 将字转换成序号
        y_data = tokenizer.encode(endText, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)

        x.append(x_data)
        y.append(y_data)
    return torch.LongTensor(x), torch.LongTensor(y)

# 加载语料
def load_corpus(corpus_path):
    corpus = ""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            corpus += line.strip()
    return corpus

# 加载词表
def load_vocab(vocab_path):
    dict = {"<padding>":0}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for inxex, line in enumerate(f):
            line = line[:-1].strip()
            dict[line] = inxex + 1
    return dict

if __name__ == '__main__':
    from config import Config
    data = load_data(Config)
    print(data)
