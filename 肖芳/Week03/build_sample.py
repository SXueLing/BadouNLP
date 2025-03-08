# 构建训练集
import json
import os
import random
import torch

def load_vocab():
  with open(os.path.join(os.path.dirname(__file__), 'vocab.json'), 'r') as f:
    vocab = json.load(f)
    return vocab


def build_dataset(count):
  vocab = load_vocab()
  X = []
  Y = []

  for i in range(count):
      x, y = build_one(vocab)
      X.append(x)
      Y.append(y)
  return torch.LongTensor(X), torch.LongTensor(Y)

# 固定生成一个6个字符的文本
def build_one(vocab):
  # 生成一个5个字符的样本 随机包含一个特殊字符, 特殊字符所在的位置是对应的分类
  unk_index = vocab["unk"]
  chars = list(vocab.keys())[1:-1]
  word = [random.choice(chars) for _ in range(5)]
  x = [vocab[c] for c in word]
  # special_char = random.choice("+-*/")
  y = random.randint(0, 5)
  # word.insert(y, special_char)
  # word = ''.join(word)
  x.insert(y, unk_index)
  return x, y

# 测试生成一个样本
# print(build_one(load_vocab()))