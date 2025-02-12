# -*- coding: utf-8 -*-

import torch
import json
import numpy as np
from model import TorchModel
from config import Config

# 读取字符到索引的映射
def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        char_to_idx = {char: idx for idx, char in enumerate(f.read().splitlines())}
    return char_to_idx

# 将句子转换为模型输入
def sentence_to_input_ids(sentence, char_to_idx, max_length):
    input_ids = [101] + [char_to_idx.get(char, 1) for char in sentence] + [102] # 未知字符映射为 1
    input_ids = input_ids[:max_length]  # 截断
    len_sentence = len(input_ids)
    attention_mask = [1] * len_sentence
    input_ids += [0] * (max_length - len_sentence)  # 补零
    attention_mask += [0] * (max_length - len_sentence)
    return (torch.tensor([input_ids], dtype=torch.long), torch.tensor([attention_mask], dtype=torch.long))

# 预测函数
def predict_sentence(sentence, model, config, char_to_idx):
    model.eval()
    input_ids, attention_mask = sentence_to_input_ids(sentence, char_to_idx, config["max_length"])
    with torch.no_grad():
        output = model(input_ids = input_ids, attention_mask = attention_mask)
    if config["use_crf"]:
        return output[0]  # CRF 返回的是解码后的标签序列
    else:
        return output

def main():
    # 加载字符映射
    char_to_idx = load_vocab("bert-base-chinese/vocab.txt")

    # 加载模型
    model = TorchModel(Config)
    model.load_state_dict(torch.load("model_output/epoch_20.pth"))
    model.eval()

    while True:
        sentence = input("请输入句子 (输入 'exit' 退出): ")
        if sentence.lower() == "exit":
            break
        pred_labels = predict_sentence(sentence, model, Config, char_to_idx)
        print(f"预测标签: {pred_labels}")

if __name__ == "__main__":
    main()
