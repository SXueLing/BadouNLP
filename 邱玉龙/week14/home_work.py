#  -*- coding: utf-8 -*-
"""
Author: loong
Time: 2025/3/13 22:08
File: home_work.py
Software: PyCharm
"""


text = "输入和输出具有不同的模态（例如文本到图像、图像到文本）"
text_encode = text.encode("utf-8")
print(len(text_encode))
tokens = list(map(int, text_encode))
print(tokens)
print(len(tokens))


def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    print(counts)
    return counts
def merge(ids, pair, idx):
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids


vocab_size = 276 # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
num_merges = vocab_size - 256

merges = {} # (int, int) -> int
for i in range(num_merges):
  stats = get_stats(tokens)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  # print(f"merging {pair} into a new token {idx}")
  ids = merge(tokens, pair, idx)
  print(ids)
  merges[pair] = idx
# print(merges)


vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
print(vocab)
def decodes(ids):
  # given ids (list of integers), return Python string
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text


def encodes(text):
  # given a string, return list of integers (the tokens)
  tokens = list(text.encode("utf-8"))
  while len(tokens) >= 2:
    stats = get_stats(tokens)
    pair = min(stats, key=lambda p: merges.get(p, float("inf")))
    if pair not in merges:
      break # nothing else can be merged
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
  return tokens

print("============")
# 输入测试文本 进行encode编码
input_st = "输入"
encode_text = list(input_st.encode("utf-8"))
# 遍历从词库获取，然后decode 看看跟原本的输入是否一致
st = b""
for i in encode_text:
  st += vocab.get(i)
print(st)
print(input_st == st.decode("utf-8"))
