from collections import defaultdict

corpus_path = "corpus.txt"
vocab_size = 300


def load_corpus(corpus_path):
    with open(corpus_path, encoding="utf8") as f:
        corpus = f.read()
    return corpus


def get_pairs(corpus_utf8):
    pairs = defaultdict(int)
    for i, j in zip(corpus_utf8, corpus_utf8[1:]):
        pairs[(i, j)] += 1
    return pairs


def init_vocab():
    vocab = {idx: bytes([idx]) for idx in range(256)}
    return vocab


def merge(list_utf8, target_pair, idx):
    new_ids = []
    i = 0
    while i < len(list_utf8):
        if i < len(list_utf8) - 1 and list_utf8[i] == target_pair[0] and list_utf8[i + 1] == target_pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(list_utf8[i])
            i += 1
    return new_ids

def decode(ids, vocab):
    tokens = b"".join(vocab[i] for i in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text

def encode(strs, merges):
    tokens = list(strs.encode("utf-8"))
    while len(tokens) > 1:
        pairs = get_pairs(tokens)
        pair = min(pairs, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        else:
            idx = merges[pair]
            tokens = merge(tokens, pair, idx)
    return tokens



corpus = load_corpus(corpus_path)
corpus_utf8 = corpus.encode("utf-8")
corpus_utf8 = list(map(int, corpus_utf8))
merges = {}
for i in range(vocab_size - 256):
    pairs = get_pairs(corpus_utf8)
    target_pair = max(pairs, key=pairs.get)
    merges[target_pair] = 256 + i
    corpus_utf8 = merge(corpus_utf8, target_pair, 256 + i)
vocab = init_vocab()
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
print(vocab)
print(merges)
input_ids = encode("我是通义千问，一个专门响应人类指令的大模型。我是效率助手，也是点子生成机，我服务于人类，致力于让生活更美好。", merges)
print(input_ids)
text = decode(input_ids, vocab)
print(text)


# vocab = {idx: bytes([idx]) for idx in range(256)}
# print(vocab)
# print(vocab[1] + vocab[2])
