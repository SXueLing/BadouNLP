import torch    
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel

class LanguageModel(nn.Module):    
    def __init__(self, bert_path, vocab):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.classify = nn.Linear(768, len(vocab))  # BERT 隐藏层维度是768
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        # x 现在是 token_ids
        bert_output = self.bert(x)[0]  # 获取BERT的最后一层输出
        bert_output = self.dropout(bert_output)
        y_pred = self.classify(bert_output)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)

#加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>":0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index + 1 #留出0位给pad token
    return vocab

#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  #输入输出错开一位
    # print(window, target)
    x = [vocab.get(word, vocab["<UNK>"]) for word in window]   #将字转换成序号
    y = [vocab.get(word, vocab["<UNK>"]) for word in target]
    return x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model

#文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
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


#计算文本ppl
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
    epoch_num = 20
    batch_size = 16       # 减小batch_size，因为BERT需要更多显存
    train_sample = 50000
    window_size = 10
    bert_path = r"/Users/cagiant/Downloads/第六周 语言模型/bert-base-chinese"
    
    # 初始化BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    vocab = build_vocab("/Users/cagiant/projects/private/DL/week10/lstm语言模型生成文本/vocab.txt")
    reverse_vocab = {v: k for k, v in vocab.items()}
    corpus = load_corpus(corpus_path)
    model = LanguageModel(bert_path, vocab)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 使用更小的学习率
    optim = torch.optim.Adam(model.parameters(), lr=2e-5)
    print("BERT模型加载完毕，开始训练")
    
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus)
            
            # 将输入转换为BERT的输入格式
            x_text = []
            for seq in x.tolist():
                text = ''
                for idx in seq:
                    if idx in reverse_vocab:
                        text += reverse_vocab[idx]
                x_text.append(text)
            # 使用BERT tokenizer处理文本
            encoded = tokenizer(
                x_text,
                padding=True,
                truncation=True,
                max_length=window_size,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                y = y.cuda()
            
            optim.zero_grad()
            loss = model(input_ids, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
            
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("罗峰祭出了星辰塔", model, vocab, window_size))
        print(generate_sentence("罗峰静下心来", model, vocab, window_size))
        
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return

if __name__ == "__main__":
    train("/Users/cagiant/projects/private/DL/week10/lstm语言模型生成文本/corpus-tun.txt", False)