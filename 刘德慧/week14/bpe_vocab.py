from config import Config


# 统计输入列表中相邻元素对的出现次数
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    将输入列表 ids 中相邻的元素对 pair 合并为一个新的元素 idx。
    :param ids: 输入的整数列表，代表一系列的 token。
    :param pair: 要合并的相邻元素对，是一个包含两个整数的元组。
    :param idx: 合并后的新元素的索引。
    :return: 合并后的新整数列表。
    """
    newids = []
    i = 0
    for i in range(len(ids)):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 1  # 跳过下一个元素
        else:
            newids.append(ids[i])
    return newids


def decode(ids):
    # given ids (list of integers), return Python string
    # 将生成器表达式中的所有字节对象拼接成一个大的字节对象 tokens
    tokens = b"".join(vocab[idx] for idx in ids)
    # bytes.decode将字节对象解码为字符串（str 类型）
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


file_path = Config['merged_file_path']
with open(file_path, 'rb') as input_file:
    text = input_file.read()  # 读取文件内容为字节对象

tokens = list(map(int, text))  # 将字节对象转换为整数列表

print('---')
print("text length:", len(text))
print('---')
print("tokens length:", len(tokens))

vocab_size = Config[
    "vocab_size"]  # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
num_merges = vocab_size - 256
ids = list(
    tokens)  # copy so we don't destroy the original list  浅拷贝：复制一份，以免破坏原始列表

merges = {}  # (int, int) -> int
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    print(f"merging {pair} into a new token {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx

# 初始化基本词汇表，将 0 到 255 的整数索引映射到对应的单字节字节对象
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    # 合并 p0 和 p1 对应的字节对象
    vocab[idx] = vocab[p0] + vocab[p1]

print('---')
print("vocab length:", len(vocab))

stats = get_stats(tokens)
merge_lst = sorted(((v, k) for k, v in stats.items()),
                   reverse=True)[:num_merges]

decode_lst = []
for item in merge_lst:
    decode_lst.append(item[1][0])

print('---')
print("decode_lst length:", len(decode_lst))
print(decode_lst)
print(decode([decode_lst]))
