import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
import json
import math

"""
基于pytorch的bert语言模型  实现SFT
"""


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        if y is not None:
            # 训练时需要mask
            mask = self.generate_mask(x)
            x, _ = self.bert(x, attention_mask = mask)        #output shape:(batch_size, sen_len, input_dim)
            y_pred = self.classify(x)   #output shape:(batch_size, sen_len, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # 预测时不需要mask
            x, _ = self.bert(x)        #output shape:(batch_size, sen_len, input_dim)
            y_pred = self.classify(x)   #output shape:(batch_size, sen_len, vocab_size)
            return torch.softmax(y_pred, dim=-1)
    
    def generate_mask(self, x):
        batch_size, seq_len = x.shape
        mask = torch.tril(torch.ones((seq_len, seq_len)), diagonal=0)  # 下三角矩阵
        s1_len = 21
        # mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
        # 前s1行只能关注前s1列
        mask[:s1_len, :s1_len] = 1
        # 从s2开始, 每一行可以关注到当前列及之前的列
        for i in range(s1_len, x.shape[1]):
            mask[i, :i+1] = 1
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # 扩展到 (batch_size, seq_len, seq_len)
        if torch.cuda.is_available():
            mask = mask.cuda()
        return mask

class DataGenerator:
    def __init__(self, data_path, tokenizer):
        self.path = data_path
        self.tokenizer = tokenizer
        self.title_max_len = 20
        self.content_max_len = 100
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                title = line["title"]
                content = line["content"]
                self.prepare_data(title, content)
        return

    #输入输出转化成序列
    def prepare_data(self, title, content):
        # input_seq = title + "<sep>" + content
        # target_seq = content + "<eos>"
        title_vec = self.tokenizer.encode(title, add_special_tokens=False, max_length=self.title_max_len, padding="max_length", truncation=True)
        content_vec = self.tokenizer.encode(content, add_special_tokens=False, max_length=self.content_max_len, padding="max_length", truncation=True)
        input_vec = title_vec + [self.tokenizer.sep_token_id] + content_vec
        target = [self.tokenizer.pad_token_id] * self.title_max_len + content_vec + [self.tokenizer.eos_token_id]
        self.data.append([torch.LongTensor(input_vec),
                          torch.LongTensor(target)])
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


#加载数据
def load_data(data_path, tokenizer, batch_size, shuffle=True):
    dg = DataGenerator(data_path, tokenizer)
    dl = DataLoader(dg, batch_size=batch_size, shuffle=shuffle)
    return dl


#建立模型
def build_model(vocab_size, hidden_size, pretrain_model_path):
    model = LanguageModel(hidden_size, vocab_size, pretrain_model_path)
    # model.bert.resize_token_embeddings(vocab_size)
    num_added_tokens = vocab_size - model.bert.config.vocab_size
    # 获取现有嵌入的均值和协方差矩阵
    with torch.no_grad():
        # 获取现有的嵌入矩阵
        model.bert.resize_token_embeddings(vocab_size)
        # model.resize_token_embeddings(len(tokenizer))  # 调整模型的嵌入矩阵大小
        old_embeddings = model.bert.embeddings.word_embeddings.weight[:-num_added_tokens, :]  # 获取旧的嵌入矩阵
        new_embeddings = model.bert.embeddings.word_embeddings.weight[-num_added_tokens:, :]  # 获取新词的嵌入矩阵

        # 计算旧嵌入的平均值
        avg_embedding = torch.mean(old_embeddings, dim=0, keepdim=True)  # 计算平均嵌入

        # 使用平均嵌入初始化新词嵌入
        with torch.no_grad():
            new_embeddings.copy_(avg_embedding)
    return model

#文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    model.eval()
    with torch.no_grad():
        pred_char = "<sep>"
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "<eos>" and len(openings) <= 100:
            openings += pred_char
            x = tokenizer.encode(openings, add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = "".join(tokenizer.decode(index))
    return openings

def sampling_strategy(prob_distribution):
    return int(torch.argmax(prob_distribution))
    # if random.random() > 0.1:
    #     strategy = "greedy"
    # else:
    #     strategy = "sampling"

    # if strategy == "greedy":
    #     return int(torch.argmax(prob_distribution))
    # elif strategy == "sampling":
    #     prob_distribution = prob_distribution.cpu().numpy()
    #     return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


def train(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 2       #每次训练样本个数
    hidden_size = 768     #bert隐层维度
    vocab_size = 21128       #bert的字表长度
    learning_rate = 0.0002    #学习率

    pretrain_model_path = "D:/NLP/pretrain_model/bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    tokenizer.add_special_tokens({"sep_token": "<sep>", "eos_token": "<eos>"})
    vocab_size = len(tokenizer)

    data = load_data(corpus_path, tokenizer, batch_size)
    model = build_model(vocab_size, hidden_size, pretrain_model_path)    #建立模型

    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    print("文本词表模型加载完毕，开始训练")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_data in data:
            x, y = batch_data
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("艺术品金融如何持续发展？", model, tokenizer))
        print(generate_sentence("你好吗？", model, tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    train("sample_data.json", False)
