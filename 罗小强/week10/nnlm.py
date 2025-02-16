import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel
from transformers import BertTokenizer

class LanguageModel(nn.Module):
    def __init__(self, input_dim, tokenizer):
        super(LanguageModel, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-chinese', return_dict=False)
        self.hidden_size = self.encoder.config.hidden_size
        self.classify = nn.Linear(self.hidden_size, tokenizer.vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, attention_mask=None, y=None):
        x = self.encoder(x, attention_mask=attention_mask)[0]
        y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)

def build_vocab():
    return BertTokenizer.from_pretrained('bert-base-chinese')

def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位
    encoding = tokenizer.encode_plus(
        window,
        padding="max_length",
        max_length=window_size,
        truncation=True,
        return_tensors="pt"
    )
    x = encoding['input_ids'].squeeze().tolist()  # 将张量转换为列表
    attention_mask_x = encoding['attention_mask'].squeeze().tolist()  # 将张量转换为列表
    encoding = tokenizer.encode_plus(
        target,
        padding="max_length",
        max_length=window_size,
        truncation=True,
        return_tensors="pt"
    )
    y = encoding['input_ids'].squeeze().tolist()  # 将张量转换为列表
    attention_mask_y = encoding['attention_mask'].squeeze().tolist()  # 将张量转换为列表
    return x, y, attention_mask_x, attention_mask_y

def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_x = []
    dataset_y = []
    attention_masks_x = []
    attention_masks_y = []
    for i in range(sample_length):
        x, y, attention_mask_x, attention_mask_y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
        attention_masks_x.append(attention_mask_x)
        attention_masks_y.append(attention_mask_y)
    return (
        torch.LongTensor(dataset_x),
        torch.LongTensor(dataset_y),
        torch.LongTensor(attention_masks_x),
        torch.LongTensor(attention_masks_y)
    )

def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model

def generate_sentence(openings, model, tokenizer, window_size):
    reverse_vocab = {v: k for k, v in tokenizer.get_vocab().items()}
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            encoding = tokenizer.encode_plus(
                openings[-window_size:],
                padding="max_length",
                max_length=window_size,
                truncation=True,
                return_tensors="pt"
            )
            x = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            if torch.cuda.is_available():
                x, attention_mask = x.cuda(), attention_mask.cuda()
            y = model(x, attention_mask=attention_mask)[0][-1]
            index = sampling_strategy(y)
            pred_char = reverse_vocab[index]
    return openings

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)

def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * ( -1 / len(sentence)))

def train(corpus_path, save_weight=True):
    epoch_num = 20        # 训练轮数
    batch_size = 64       # 每次训练样本个数
    train_sample = 50000   # 每轮训练总共训练的样本总数
    char_dim = 256        # 每个字的维度
    window_size = 10       # 样本文本长度
    tokenizer = build_vocab()       # 建立字表
    corpus = load_corpus(corpus_path)     # 加载语料
    model = build_model(tokenizer, char_dim)    # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)   # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y, attention_mask_x, attention_mask_y = build_dataset(batch_size, tokenizer, window_size, corpus) # 构建一组训练样本
            if torch.cuda.is_available():
                x, y, attention_mask_x, attention_mask_y = x.cuda(), y.cuda(), attention_mask_x.cuda(), attention_mask_y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, attention_mask=attention_mask_x, y=y)
            loss.backward()      # 计算梯度
            optim.step()         # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return

if __name__ == "__main__":
    train("corpus.txt", False)
