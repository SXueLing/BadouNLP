import os
import re


def concat_txt(root_dir):
    text = ""
    # 遍历目录树
    for root, _, files in os.walk(root_dir):
        for filename in sorted(files):  # 按文件名排序保证顺序一致性
            if filename.endswith('.txt'):
                filepath = os.path.join(root, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    text += re.sub(r'\n|\r|\t|<br>', '', f.read())
    return text


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


def create_vocab(tokens):
    vocab_size = 276  # 超参数：预期的最终词表大小
    num_merges = vocab_size - 256
    ids = list(tokens)
    merges = {}  # (int, int) -> int
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        # print(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx

    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]

    return ids, merges, vocab


def decode(ids, vocab):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text


def encode(text, merges):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens


# 使用示例
if __name__ == '__main__':
    text = concat_txt("Heroes")
    print(f"\n拼接结果（总长度：{len(text)} 字符）：\n{text[:500]}...")
    tokens = text.encode("utf-8")
    tokens = list(map(int, tokens))
    ids, merges, vocab = create_vocab(tokens)
    print("----------------------------")
    print("tokens length:", len(tokens))
    print("after bpe, tokens length:", len(ids))
    print(f"compression ratio: {len(tokens) / len(ids):.2f}X")
    print("merges:", merges)
    print("----------------------------")
    text_test = "这个世界是如何形成的"
    text_encode = encode(text_test, merges)
    print("编码一句话：", text_encode)
    print("再解码：", decode(text_encode, vocab))
