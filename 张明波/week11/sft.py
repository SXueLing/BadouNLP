# coding:utf8
import json
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random


class EnhancedBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids, attention_mask=None, prompt_lengths=None):
        # 自动生成增强型注意力掩码
        if prompt_lengths is not None and attention_mask is None:
            attention_mask = self.build_enhanced_mask(input_ids, prompt_lengths)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False
        )

    @staticmethod
    def build_enhanced_mask(input_ids, prompt_lengths):
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # 基础因果掩码
        mask = torch.tril(torch.ones(batch_size, seq_len, seq_len, device=device))

        # 增强段落感知
        for i, p_len in enumerate(prompt_lengths):
            if p_len > 0:
                # Prompt区域完全可见
                mask[i, :p_len, :p_len] = 1
                # Answer区域可以看见整个Prompt
                mask[i, p_len:, :p_len] = 1
        return mask


class SFTLanguageModel(nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        config = BertConfig.from_pretrained(pretrain_path)
        self.bert = EnhancedBertModel.from_pretrained(pretrain_path, config=config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None, prompt_lengths=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths
        )
        sequence_output = outputs[0]
        logits = self.lm_head(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            return loss
        return logits


class SFDDataset(Dataset):
    def __init__(self, tokenizer, corpus, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.process_corpus(corpus)

    def process_corpus(self, corpus):
        processed = []
        for prompt, answer in corpus:
            # 编码时自动添加特殊标记
            prompt_enc = self.tokenizer.encode(prompt, add_special_tokens=False)
            answer_enc = self.tokenizer.encode(answer, add_special_tokens=False)

            # 构建模型输入
            input_ids = [self.tokenizer.cls_token_id] + prompt_enc + \
                        [self.tokenizer.sep_token_id] + answer_enc + \
                        [self.tokenizer.sep_token_id]

            # 构建标签（-100表示忽略预测）
            labels = [-100] * (len(prompt_enc) + 2) + answer_enc + [self.tokenizer.sep_token_id]

            # 记录prompt长度（用于自动生成mask）
            prompt_length = len(prompt_enc) + 2  # CLS + prompt + SEP

            # 截断填充处理
            pad_len = self.max_length - len(input_ids)
            if pad_len > 0:
                input_ids += [self.tokenizer.pad_token_id] * pad_len
                labels += [-100] * pad_len
            else:
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]

            processed.append({
                "input_ids": torch.LongTensor(input_ids),
                "labels": torch.LongTensor(labels),
                "prompt_length": prompt_length
            })
        return processed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def sft_collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "prompt_lengths": torch.LongTensor([x["prompt_length"] for x in batch])
    }


def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.9):
    model.eval()
    device = next(model.parameters()).device

    # 编码prompt
    prompt_enc = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = [tokenizer.cls_token_id] + prompt_enc + [tokenizer.sep_token_id]
    prompt_length = len(input_ids)

    for _ in range(max_length):
        inputs = torch.LongTensor([input_ids]).to(device)
        with torch.no_grad():
            outputs = model(inputs, prompt_lengths=[prompt_length])
        next_token_logits = outputs[0, -1, :]

        # 温度采样
        next_token_logits = next_token_logits / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        if next_token == tokenizer.sep_token_id:
            break

        input_ids.append(next_token)

    return tokenizer.decode(input_ids[prompt_length:], skip_special_tokens=True)


def main():
    # 配置参数
    config = {
        "pretrain_path": "bert-base-chinese",
        "data_path": "data/sft_samples.json",
        "max_length": 256,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "num_epochs": 10,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # 初始化组件
    tokenizer = BertTokenizer.from_pretrained(config["pretrain_path"])
    model = SFTLanguageModel(config["pretrain_path"]).to(config["device"])

    # 加载数据
    with open(config["data_path"]) as f:
        corpus = [json.loads(line) for line in f]
    dataset = SFDDataset(tokenizer, corpus, config["max_length"])
    dataloader = DataLoader(dataset,
                            batch_size=config["batch_size"],
                            collate_fn=sft_collate_fn,
                            shuffle=True)

    # 训练准备
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    # 训练循环
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0

        for batch in dataloader:
            inputs = batch["input_ids"].to(config["device"])
            labels = batch["labels"].to(config["device"])
            prompt_lengths = batch["prompt_lengths"].to(config["device"])

            optimizer.zero_grad()
            loss = model(inputs, labels=labels, prompt_lengths=prompt_lengths)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 验证生成
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print(f"Avg Loss: {total_loss / len(dataloader):.4f}")
        test_prompts = ["近期股市走势", "人工智能的未来发展"]
        for prompt in test_prompts:
            generated = generate_text(model, tokenizer, prompt)
            print(f"Prompt: {prompt}\nGenerated: {generated}\n")

    # 保存模型
    output_dir = "sft_model"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
