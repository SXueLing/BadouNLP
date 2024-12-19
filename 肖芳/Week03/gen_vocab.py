import json
import os

# 生成词表
chars = "abcdefghijklmnopqrstuvwxyz"
vocab = {"pad": 0}
for index,char in enumerate(chars):
  vocab[char] = index+1
vocab["unk"] = len(chars) + 1

print('vocab', vocab)

with open(os.path.join(os.path.dirname(__file__), 'vocab.json'), 'w') as f:
  json.dump(vocab, f, indent=4)
