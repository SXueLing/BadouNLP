import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
from transformers import BertModel, BertTokenizer


class BertQA_Model(nn.Module):
    def __init__(self, bert_model_path, vocab_size):
        super(BertQA_Model, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
        self.classify = nn.Linear(self.bert.config.hidden_size, vocab_size)  # 输出词汇表大小的预测
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, attention_mask=None, y=None):
        output = self.bert(input_ids=x, attention_mask=attention_mask)
        last_hidden_state = output[0]  # [batch_size, seq_len, hidden_dim]
        logits = self.classify(last_hidden_state)  # [batch_size, seq_len, vocab_size]

        if y is not None:
            return self.loss(logits.view(-1, logits.shape[-1]), y.view(-1))  # 计算交叉熵损失
        else:
            return torch.softmax(logits, dim=-1)  # 返回概率分布


def build_qa_sample_from_json(data, vocab, window_size):
    sample = random.choice(data)
    title = sample['title']
    content = sample['content']
    question = title
    answer = content
    x = [vocab.get(word, vocab["<UNK>"]) for word in question]
    y = [0] * len(x)

    for i, word in enumerate(answer):
        if word in question:
            idx = min(i, len(y) - 1)
            y[idx] = 1
    x = x[:window_size]
    y = y[:window_size]

    while len(x) < window_size:
        x.append(vocab["<pad>"])
    while len(y) < window_size:
        y.append(0)

    return x, y

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

# 加载词汇表
def build_vocab(vocab_path):
    vocab = {"<pad>": 0, "<UNK>": 1}  # 默认设置 pad 和 unk token
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()  # 去掉换行符
            vocab[char] = index + 2  # 从2开始分配ID（保留0和1给<pad>和<UNK>）
    return vocab

# 构建数据集
def build_qa_dataset_from_json(sample_length, vocab, window_size, data):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_qa_sample_from_json(data, vocab, window_size)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def train(corpus_path, vocab_path, bert_model_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    window_size = 20
    vocab = build_vocab(vocab_path)  # 建立字表
    data = load_json_data(corpus_path)  # 加载JSON数据
    vocab_size = len(vocab)
    model = BertQA_Model(bert_model_path, vocab_size)
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)  # 初始化tokenizer

    if torch.cuda.is_available():
        model = model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=1e-5)  # 建立优化器
    print("模型加载完毕，开始训练")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_qa_dataset_from_json(batch_size, vocab, window_size, data)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            attention_mask = (x != 0).float()  # attention mask
            optim.zero_grad()
            loss = model(x, attention_mask=attention_mask, y=y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("json", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    train("./sample_data.json", "./vocab.txt", r"D:\AI\homeWork\pretrain_models\bert-base-chinese", save_weight=True)
