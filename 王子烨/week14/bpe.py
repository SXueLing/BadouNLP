# -*- coding: utf-8 -*-
# @Time    : 2025/3/12 17:04
# @Author  : yeye
# @File    : bpe.py
# @Software: PyCharm
# @Desc    :
with open(r"D:\code\pycharm\NLP\week14\work\output\target.txt", encoding='utf-8') as f:
    text = f.read()

print(text)
tokens = text.encode("utf-8")
tokens = list(map(int, tokens))


# print(tokens)


def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


# ---
vocab_size = 600  # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
num_merges = vocab_size - 256
ids = list(tokens)  # copy so we don't destroy the original list

merges = {}  # (int, int) -> int
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    print(f"merging {pair} into a new token {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx

# print("tokens length:", len(tokens))
# print("ids length:", len(ids))
# print(f"compression ratio: {len(tokens) / len(ids):.2f}X")
# print(merges)


vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]


# print(vocab)


def decode(ids):
    # given ids (list of integers), return Python string
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text


def encode(text):
    # given a string, return list of integers (the tokens)
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # nothing else can be merged
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens


# text2 = decode(encode(text))
# print(text2 == text)
# print(len(tokens))
# print(len(encode(text)))
print(merges)
i = 0
for pair, k in merges.items():
    if i<350:
        token = list(pair)
        token = decode(token)
        print(token)
        i = i + 2
    else:
        break
