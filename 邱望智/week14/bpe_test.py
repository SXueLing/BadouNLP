import os

'''
bpe构建词表测试
'''


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


def build_vocab(text):
    vocab_size = 500
    num_merges = vocab_size - 256
    tokens = text.encode("utf-8")  # raw bytes
    tokens = list(map(int, tokens))
    ids = list(tokens)

    merges = {}  # (int, int) -> int
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        ids = merge(ids, pair, idx)
        merges[pair] = idx

    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
        try:
            print(idx, vocab[idx].decode("utf8"))
        except UnicodeDecodeError:
            continue

    return merges, vocab


# 编码过程
def decode(ids, vocab):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text


# 解码过程
def encode(text, merges):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # nothing else can be merged
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens


if __name__ == "__main__":
    dir_path = "dota2英雄介绍-byRAG\Heroes"
    # 所有文件读成一个长字符串。也可以试试只读入一个文件
    corpus = ""
    for path in os.listdir(dir_path):
        path = os.path.join(dir_path, path)
        with open(path, encoding="utf8") as f:
            text = f.read()
            corpus += text + '\n'
    # 构建词表
    merges, vocabs = build_vocab(corpus)
    # 使用词表进行编解码
    string = "冥界亚龙"
    encode_ids = encode(string, merges)
    print("编码结果：", encode_ids)
    decode_string = decode(encode_ids, vocabs)
    print("解码结果：", decode_string)
