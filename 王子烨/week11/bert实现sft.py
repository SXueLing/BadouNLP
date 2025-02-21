# coding:utf8
import json
import torch
import torch.nn as nn
import numpy as np
import random
import os
from transformers import BertModel, BertTokenizer

"""
基于 PyTorch + BERT 的语言模型，
用标题 (title) 预测内容 (content) 进行 SFT 训练，
输入为标题和内容拼接后的序列，
并构造自定义注意力掩码矩阵：
  - 标题-标题部分：全1
  - 标题-内容部分：全0
  - 内容-标题部分：全1
  - 内容-内容部分：下三角矩阵
"""


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(r'E:\model\bert-base-chinese')
        self.classify = nn.Linear(768, len(vocab))
        self.dropout = nn.Dropout(0.1)
        self.loss_fn = nn.CrossEntropyLoss()
        # 固定标题和内容的最大长度（与 prepare_data 中保持一致）
        self.title_length = 20
        self.content_length = 20
        self.total_length = self.title_length + self.content_length

    def generate_sft_mask(self, batch_size, device):
        """
        生成 SFT 用的注意力掩码，形状为 (batch_size, total_length, total_length)：
          - 左上 (title-title)：全1
          - 右上 (title-content)：全0
          - 左下 (content-title)：全1
          - 右下 (content-content)：下三角矩阵
        """
        title_len = self.title_length
        content_len = self.content_length
        # 标题-标题：全1
        title_mask = torch.ones(title_len, title_len, device=device, dtype=torch.long)
        # 标题-内容：全0
        title_to_content = torch.zeros(title_len, content_len, device=device, dtype=torch.long)
        # 内容-标题：全1
        content_to_title = torch.ones(content_len, title_len, device=device, dtype=torch.long)
        # 内容-内容：下三角
        content_mask = torch.tril(torch.ones(content_len, content_len, device=device, dtype=torch.long))

        top = torch.cat([title_mask, title_to_content], dim=1)  # (title_len, total_length)
        bottom = torch.cat([content_to_title, content_mask], dim=1)  # (content_len, total_length)
        full_mask = torch.cat([top, bottom], dim=0)  # (total_length, total_length)

        # 扩展 batch 维度
        return full_mask.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, x, y=None):
        # x 的形状：(batch_size, total_length)
        batch_size, seq_len = x.shape  # seq_len 应等于 total_length
        attention_mask = self.generate_sft_mask(batch_size, x.device)
        outputs = self.bert(input_ids=x, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, total_length, 768)
        logits = self.classify(self.dropout(last_hidden_state))  # (batch_size, total_length, vocab_size)
        if y is not None:
            return self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        else:
            return torch.softmax(logits, dim=-1)


def build_vocab(bert_path):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    vocab = tokenizer.get_vocab()
    return vocab, tokenizer


def load_corpus(path):
    corpus = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                corpus.append(json.loads(line))
            except json.decoder.JSONDecodeError as e:
                print(f"JSON decode error: {e} in line: {line}")
    return corpus


def build_dataset(tokenizer, corpus, title_length=20, content_length=20):
    dataset_x, dataset_y = [], []
    for line in corpus:
        title = line["title"]
        content = line["content"]
        tokens, labels = prepare_data(tokenizer, title, content, title_length, content_length)
        dataset_x.append(tokens)
        dataset_y.append(labels)
    return torch.stack(dataset_x), torch.stack(dataset_y)


def prepare_data(tokenizer, title, content, title_length, content_length):
    # 对标题和内容分别进行编码
    title_encoding = tokenizer(title, return_tensors="pt", padding="max_length", truncation=True,
                               max_length=title_length)
    content_encoding = tokenizer(content, return_tensors="pt", padding="max_length", truncation=True,
                                 max_length=content_length)
    title_ids = title_encoding["input_ids"].squeeze(0)  # (title_length,)
    content_ids = content_encoding["input_ids"].squeeze(0)  # (content_length,)
    # 拼接形成整体输入（标题在前，内容在后）
    input_ids = torch.cat([title_ids, content_ids], dim=0)  # (title_length+content_length,)
    # 构造标签：标题部分不计算损失，用 -100 屏蔽；内容部分为实际 tokens
    ignore_labels = torch.full((title_length,), -100, dtype=torch.long)
    labels = torch.cat([ignore_labels, content_ids], dim=0)
    return input_ids, labels


def build_model(vocab, char_dim):
    return LanguageModel(char_dim, vocab)


def generate_sentence(openings, model, tokenizer, window_size=40):
    """
    以 openings（标题）为开头，生成内容。
    注意：此处生成时输入为固定长度 window_size（标题+内容），后续 token 按生成策略补充。
    """
    model.eval()
    with torch.no_grad():
        while True:
            tokens = tokenizer(openings, return_tensors="pt", padding="max_length",
                               truncation=True, max_length=window_size)
            x = tokens["input_ids"]
            if torch.cuda.is_available():
                x = x.cuda()
            logits = model(x)
            # 取最后一个 token 的预测分布
            last_token_logits = logits[0, -1, :]
            index = sampling_strategy(last_token_logits)
            pred_token = tokenizer.decode([index])
            if pred_token == "[SEP]" or len(openings.split()) > 30:
                break
            openings += pred_token
    return openings


def sampling_strategy(prob_distribution):
    # 根据随机策略选择：贪心或采样
    if random.random() > 0.1:
        return int(torch.argmax(prob_distribution))
    else:
        prob_distribution = prob_distribution.cpu().numpy()
        prob_distribution = prob_distribution / np.sum(prob_distribution)
        return int(np.random.choice(len(prob_distribution), p=prob_distribution))


def train(corpus_path, save_weight=True):
    epoch_num = 20
    batch_size = 64
    char_dim = 256
    vocab, tokenizer = build_vocab(r"E:\model\bert-base-chinese")
    corpus = load_corpus(corpus_path)
    model = build_model(vocab, char_dim)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)
    print("加载模型和数据，开始训练")
    x, y = build_dataset(tokenizer, corpus, title_length=20, content_length=20)
    dataset_size = x.size(0)
    permutation = torch.randperm(dataset_size)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i: i + batch_size]
            batch_x = x[indices]
            batch_y = y[indices]
            if torch.cuda.is_available():
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            optim.zero_grad()
            loss = model(batch_x, batch_y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print(f"Epoch {epoch + 1}, Loss: {np.mean(watch_loss):.4f}")
    if save_weight:
        os.makedirs("model", exist_ok=True)
        model_path = os.path.join("model", os.path.basename(corpus_path).replace("txt", "pth"))
        torch.save(model.state_dict(), model_path)
    print("训练完成。")


def test_model(corpus_path):
    """
    测试案例：加载训练好的模型权重，给定一个示例标题，生成对应的内容预测
    """
    vocab, tokenizer = build_vocab(r"E:\model\bert-base-chinese")
    model = build_model(vocab, 256)
    model_path = os.path.join("model", os.path.basename(corpus_path).replace("txt", "pth"))
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
        model = model.cuda()
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    sample_title = "阿根廷歹徒抢服装尺码不对拿回店里换"
    # 这里 openings 仅用标题作为生成起点
    generated = generate_sentence(sample_title, model, tokenizer, window_size=40)
    print("输入标题:", sample_title)
    print("生成内容:", generated)


if __name__ == "__main__":
    # 训练模型（若已训练，可设置 save_weight=False 直接测试）
    train(r"D:\code\pycharm\NLP\week11\bert语言模型生成文本\sample_data.json", save_weight=True)
    # 测试案例：加载权重后生成内容
    test_model(r"D:\code\pycharm\NLP\week11\bert语言模型生成文本\sample_data.json")
