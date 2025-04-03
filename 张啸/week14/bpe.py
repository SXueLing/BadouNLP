import os

"""
bpe构建词表
"""


# 统计相邻字节对的出现频率
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

# 将指定字节对pair合并为新字节idx
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
    # ids是将字符串text按UTF-8编码后，每个字节转换成0～255的整数
    tokens = text.encode("utf-8")
    tokens = list(map(int, tokens))
    ids = list(tokens)

    merges = {}
    for i in range(num_merges):
        # 统计最频繁的字节对
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        print(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx

    vocab = {idx: bytes([idx]) for idx in range(256)}
    # 更新vocab表
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
        try:
            # 将unicode编码转换为可读的字符,打印出来看一看
            print(idx, vocab[idx].decode("utf-8"))
        except UnicodeDecodeError:
            # 部分的词其实是部分unicode编码的组合，无法转译为完整的字符
            # 但是没关系，模型依然会把他们当成一整整体来理解
            continue
    
    #实际情况中，应该把这些词表记录到文件中，就可以用于未来的的编码和解码了
    #可以只存储merges,因为vocab总是可以通过merges再构建出来，当然也可以两个都存储
    return merges, vocab

def decode(ids, vocab):
    tokens = b"".join(vocab[i] for i in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text

def encode(text, merges):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        # 通过结合统计结果（stats）和预先定义的合并优先级（merges），从所有可能的 token 对中挑选出一个合适的、按照规则应先合并的 pair
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

if __name__ == "__main__":
    dir_path = "Heroes"
    corpus = ""
    for path in os.listdir(dir_path):
        path = os.path.join(dir_path, path)
        with open(path, encoding="utf-8") as f:
            text = f.read()
            corpus += text + "\n"

    # 构建词表
    merges, vocabs = build_vocab(corpus)
    # 使用词表进行编码
    string = "斧王"
    encode_ids = encode(string, merges)
    print("编码结果：", encode_ids)
    decode_string = decode(encode_ids, vocabs)
    print("解码结果：", decode_string)
