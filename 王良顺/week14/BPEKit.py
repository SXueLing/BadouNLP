# -*- coding: utf-8 -*-
# @Time    : 2025/3/10 14:02
# @Author  : WLS
# @File    : BPEKit.py
# @Software: PyCharm
from loader_txt import read_file

file_path = './Heroes/concatenated.txt'

text = read_file(file_path)

tokens = text.encode("utf-8") # raw bytes
tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience
print("text length:", len(text))
print("tokens length:", len(tokens))

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts
stats = get_stats(tokens)
# print(stats)
sort_list = sorted(((v,k) for k,v in stats.items()), reverse=True)

# 打印前50个高频词
text_stats = get_stats(text)
sort_text_stats = sorted(((v,k) for k,v in text_stats.items()), reverse=True)
for i in range(50):
  print(sort_text_stats[i])


def merge(ids, pair, idx):
  # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
  newids = []
  i = 0
  while i < len(ids):
    # if we are not at the very last position AND the pair matches, replace it
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids


vocab_size = 768 # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
num_merges = vocab_size - 256
ids = list(tokens) # copy so we don't destroy the original list

merges = {} # (int, int) -> int
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  # print(f"merging {pair} into a new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx


def encode(text):
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

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
  # given ids (list of integers), return Python string
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text

Lord_of_wolves = read_file(r"./Lord of Wolves.txt")
Lord_of_wolves_tokens = encode(Lord_of_wolves)
print(decode(Lord_of_wolves_tokens))

