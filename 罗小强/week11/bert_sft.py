import json
import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LanguageModel(nn.Module):
    def __init__(self, input_dim):
        super(LanguageModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.encoder = BertModel.from_pretrained('bert-base-chinese', return_dict=False)
        self.hidden_size = self.encoder.config.hidden_size
        self.classify = nn.Linear(self.hidden_size, self.tokenizer.vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    def forward(self, x, attention_mask=None, y=None):
        x = self.encoder(x, attention_mask=attention_mask)[0]
        y_pred = self.classify(x)  # output shape: (batch_size, seq_len, vocab_size)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


class NewsDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_length=512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.loads(f.read())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        title, content = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            title,
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer.encode_plus(
            content,
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze()
        }


def train(corpus_path, save_weight=True):
    epoch_num = 20        # 训练轮数
    batch_size = 8        # 每次训练样本个数
    char_dim = 256        # 每个字的维度
    max_seq_length = 512  # 最大序列长度

    model = LanguageModel(char_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=5e-5)  # 建立优化器
    tokenizer = BertTokenizer.from_pretrained('../week10/bert-base-chinese')

    dataset = NewsDataset(corpus_path, tokenizer, max_seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("开始训练...")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in dataloader:
            x = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            optim.zero_grad()  # 梯度归零
            loss = model(x, attention_mask=attention_mask, y=y)
            loss.backward()      # 计算梯度
            optim.step()         # 更新权重
            watch_loss.append(loss.item())

        avg_loss = np.mean(watch_loss)
        print(f"第 {epoch + 1} 轮平均 loss: {avg_loss}")

    if save_weight:
        base_name = os.path.basename(corpus_path).replace("json", "pth")
        model_path = os.path.join("model", base_name)
        if not os.path.exists("model"):
            os.mkdir("model")
        torch.save(model.state_dict(), model_path)
        print(f"模型保存到: {model_path}")
    return model


def evaluate(model, corpus_path, max_seq_length=512):
    """
    评估模型性能，计算困惑度（Perplexity）
    """
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('../week10/bert-base-chinese')
    dataset = NewsDataset(corpus_path, tokenizer, max_seq_length)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    total_loss = 0
    token_count = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            loss = model(x, attention_mask=attention_mask, y=y)
            total_loss += loss.item() * (y != tokenizer.pad_token_id).sum().item()
            token_count += (y != tokenizer.pad_token_id).sum().item()

    perplexity = math.exp(total_loss / token_count)
    print(f"评估完成，困惑度 (Perplexity): {perplexity:.4f}")
    return perplexity


def generate_text(model, prompt, max_length=50, temperature=1.0):
    """
    根据给定的提示生成文本
    """
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('../week10/bert-base-chinese')

    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated_tokens = input_ids.tolist()[0]

        for _ in range(max_length):
            attention_mask = torch.ones_like(input_ids).to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            next_token_probs = outputs[0, -1] / temperature
            next_token = torch.multinomial(next_token_probs, num_samples=1).item()

            if next_token == tokenizer.sep_token_id or next_token == tokenizer.pad_token_id:
                break

            generated_tokens.append(next_token)
            input_ids = torch.tensor([generated_tokens]).to(device)

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"生成的文本: {generated_text}")
    return generated_text


if __name__ == "__main__":
    # 训练模型
    trained_model = train("news_teacher.json", True)

    # 评估模型
    evaluate(trained_model, "news_teacher.json")

    # 预测生成文本
    prompt = "浙江宁波冬至舌尖特写：咸鲜汤圆里的海洋文化"
    generate_text(trained_model, prompt, max_length=100)