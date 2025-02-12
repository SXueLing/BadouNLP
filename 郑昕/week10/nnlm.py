import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, BertTokenizer

"""
基于pytorch的BERT语言模型
"""

class LanguageModel(nn.Module):
    def __init__(self):
        super(LanguageModel, self).__init__()
        # 加载bert-base-chinese预训练模型
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.classify = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, attention_mask=None, y=None):
        outputs = self.bert(input_ids=x, attention_mask=attention_mask)
        sequence_output = outputs[0]  # shape: (batch_size, seq_len, hidden_size)
        y_pred = self.classify(sequence_output)  # shape: (batch_size, seq_len, vocab_size)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)

# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位
    x = tokenizer.encode(window, add_special_tokens=False)
    y = tokenizer.encode(target, add_special_tokens=False)
    attention_mask = [1] * len(x)
    return x, y, attention_mask

# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# tokenizer 分词器
# window_size 样本长度
# corpus 语料字符串
def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_x = []
    dataset_y = []
    attention_masks = []
    for i in range(sample_length):
        x, y, attention_mask = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
        attention_masks.append(attention_mask)
    # 填充到相同长度
    max_len = max([len(seq) for seq in dataset_x])
    for i in range(len(dataset_x)):
        pad_len = max_len - len(dataset_x[i])
        dataset_x[i] = dataset_x[i] + [tokenizer.pad_token_id] * pad_len
        dataset_y[i] = dataset_y[i] + [tokenizer.pad_token_id] * pad_len
        attention_masks[i] = attention_masks[i] + [0] * pad_len
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y), torch.LongTensor(attention_masks)

# 建立模型
def build_model():
    model = LanguageModel()
    return model

# 文本生成测试代码
def generate_sentence(openings, model, tokenizer, window_size):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = tokenizer.encode(openings[-window_size:], add_special_tokens=False)
            attention_mask = [1] * len(x)
            x = torch.LongTensor([x])
            attention_mask = torch.LongTensor([attention_mask])
            if torch.cuda.is_available():
                x = x.cuda()
                attention_mask = attention_mask.cuda()
            y = model(x, attention_mask)[0][-1]
            index = sampling_strategy(y)
            pred_char = tokenizer.convert_ids_to_tokens([index])[0]
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

# 计算文本ppl
def calc_perplexity(sentence, model, tokenizer, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = tokenizer.encode(window, add_special_tokens=False)
            attention_mask = [1] * len(x)
            x = torch.LongTensor([x])
            attention_mask = torch.LongTensor([attention_mask])
            target = sentence[i]
            target_index = tokenizer.encode(target, add_special_tokens=False)[0]
            if torch.cuda.is_available():
                x = x.cuda()
                attention_mask = attention_mask.cuda()
            pred_prob_distribute = model(x, attention_mask)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))

def train(corpus_path, save_weight=True):
    epoch_num = 10  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    window_size = 10  # 样本文本长度
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    corpus = load_corpus(corpus_path)  # 加载语料
    model = build_model()  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y, attention_mask = build_dataset(batch_size, tokenizer, window_size, corpus)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y, attention_mask = x.cuda(), y.cuda(), attention_mask.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, attention_mask, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
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