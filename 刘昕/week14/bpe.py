text = "另外，要提醒用户如何检查当前文件的编码，比如在VS Code中查看底部状态栏，或者在命令行使用file命令。如果用户不确定如何操作，可能需要指导他们如何用记事本另存为UTF-8格式，或者推荐使用支持编码检测的编辑器。"

def get_stats(ids):
    pair_count = {}
    for pair in zip(ids, ids[1:]):
        pair_count[pair] = pair_count.get(pair, 0) + 1
    return pair_count

def merge(ids, idx, pair):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def decode(ids, vocab):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text

tokens = text.encode("utf-8") # raw bytes
tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience

tokens_num = 300
merges = {}

for idx in range(256, tokens_num):
    pair_count = get_stats(tokens)
    top_pair = max(pair_count, key = pair_count.get)
    tokens = merge(tokens, idx, top_pair)
    merges[top_pair] = idx

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

print(decode(tokens, vocab))