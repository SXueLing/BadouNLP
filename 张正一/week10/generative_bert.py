import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from transformers import BertTokenizer, BertModel

class Generator(nn.Module):
    def __init__(self, batch_size, hidden_size, max_length):
        super().__init__()
        self.bert_pretrained_path = r'C:\Users\81080\Documents\Python_Study\AI\week06\bert-base-chinese'
        self.bert = BertModel.from_pretrained(self.bert_pretrained_path, return_dict=False)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_pretrained_path)
        self.classify = nn.Linear(hidden_size, len(self.tokenizer.vocab))
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss()
        
    
    def encode_sentence(self, sentence, window_size):
        ids = self.tokenizer.encode(sentence, add_special_tokens=False, max_length = window_size, truncation=True, padding='max_length')
        return ids
    
    def build_dataset(self, batch_size, window_size, corpus):
        x = []
        y = []
        for i in range(batch_size):
            start_index = np.random.randint(0, len(corpus) - 1 - window_size)
            end_index = start_index + window_size
            x.append(self.encode_sentence(corpus[start_index:end_index], window_size))
            y.append(self.encode_sentence(corpus[start_index+1:end_index+1], window_size))
            # print(x, y)
            # print(len(x), len(y))
        return torch.LongTensor(x), torch.LongTensor(y)

    def forward(self, x, y = None):
        # print(x.shape, y.shape, y_pred.shape)
        if y is not None:
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1])))  # 下三角矩阵
            mask = mask.masked_fill(mask == 0, float('-inf'))
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.bert(x, attention_mask=mask)  # [batch_size, seq_len, hidden_size]
            y_pred = self.classify(x)  # [batch_size, seq_len, vocab_size]
            # y_pred.view(-1, len(self.tokenizer.vocab)) [batch_size * seq_len, vocab_size]  y.view(-1).shape [batch_size * seq_len]
            return self.loss(y_pred.view(-1, len(self.tokenizer.vocab)), y.view(-1))
        else:
            # 预测不用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)

def load_corpus(path):
    corpus = ''
    with open(path, 'r', encoding='gbk') as f:
        for line in f:
            corpus += line.strip()
    return corpus

def generate_sentence(model, start_sentence, window_size):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        while pred_char != "\n" and len(start_sentence) <= 30:
            start_sentence += pred_char
            input_ids = model.tokenizer.encode(start_sentence, add_special_tokens=False)
            input_ids = torch.LongTensor([input_ids])
            # print(54, input_ids)
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
            output = model(input_ids)
            # print(66, output, output.shape)
            # print(67, output[0])
            # print(68, output[0][-1])
            # print(58, output)
            output_id = sampling_strategy(output[0][-1])
            # print(60, output_ids)
            output_char = model.tokenizer.decode(output_id)
            # print(76, ''.join(output_char))
            pred_char = ''.join(output_char)
            # print(62, output_sentence)
    return start_sentence

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

def train():
    epoch_num = 20     #训练轮数
    batch_size = 128       #每次训练样本个数
    train_sample = 10000   #每轮训练总共训练的样本总数
    char_dim = 768        #每个字的维度
    window_size = 12      #样本文本长度
    corpus = load_corpus('corpus.txt')
    model = Generator(batch_size, char_dim, window_size)
    if torch.cuda.is_available():
        model = model.cuda()
    # generate_sentence(model, '李白站在山路上，说', window_size)
    # return
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(train_sample // batch_size):
            x, y = model.build_dataset(batch_size, window_size, corpus)
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        print(f'第{epoch+1}轮loss：{np.mean(watch_loss)}')
        print(generate_sentence(model, '让他在半年之前，就不能做出', window_size))
        print(generate_sentence(model, '李慕站在山路上，深深的呼吸', window_size))

if __name__ == '__main__':
    train()
