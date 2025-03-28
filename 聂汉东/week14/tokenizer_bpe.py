import os


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


def load_data(path):
    heroes = 0
    text = ""
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            file_path = os.path.join(path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text += f.read()
                heroes += 1
    print(f'收集了{heroes}个英雄资料')
    return text


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


def load_vocab(tokens, vocab_size=516):
    num_merges = vocab_size - 256
    ids = list(tokens)
    merges = {}
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        # print(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    vocab_list = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab_list[idx] = vocab_list[p0] + vocab_list[p1]
    return vocab_list, merges


if __name__ == '__main__':
    path = r'D:\workspace\pycharm-works\badou\week14 大语言模型相关第四讲\RAG\dota2英雄介绍-byRAG\Heroes'
    text = load_data(path)
    tokens = text.encode('utf-8')
    tokens = list(map(int, tokens))
    vocab, merges = load_vocab(tokens)
    valtext = "你好我只是测试一下"
    print("encode:", encode(valtext))
    print("decode:", decode(encode(valtext)))
    print(decode(encode(valtext)) == valtext)
