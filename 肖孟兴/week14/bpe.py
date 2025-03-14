def build_token(corpus_path):
    # 初始化一个空字符串，用于存储所有读取的文本
    text = ''
    # 使用with语句打开文件，确保文件在使用后会被正确关闭
    # corpus_path是文件的路径，encoding="utf8"指定文件编码为UTF-8
    with open(corpus_path, encoding="utf8") as f:
        # 使用enumerate函数遍历文件的每一行，index是行号，line是当前行的内容
        for index, line in enumerate(f):
            # 去掉换行符
            line = line.strip()
            text = text + line
    token = text.encode("utf-8")
    token = list(map(int, token))
    return token
def get_stats(ids):
    # 初始化一个空字典，用于存储ID对的出现次数
    counts = {}
    # 使用zip函数将ids列表与其自身从第二个元素开始的子列表配对
    # 这样可以生成所有相邻ID对的组合
    for pair in zip(ids, ids[1:]):
        # 如果该ID对已经在字典中，则将其计数加1
        # 如果不在字典中，则初始化为1
        counts[pair] = counts.get(pair, 0) + 1
    # 返回包含所有ID对及其出现次数的字典
    return counts

def merge(ids, pair, idx):
    new_ids = []
    # 只能用while循环，不能用for循环，因为for循环的迭代变量是从range里提取的，不能改变
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def build_merges(ids, num_merges):
    tokens = ids
    merges = {}
    for i in range(num_merges):
        # 获取当前token的统计信息
        stats = get_stats(ids)
        # 找到出现次数最多的ID对
        pair = max(stats, key=stats.get)
        # 将该ID对合并为一个新ID
        idx = 256 + i
        # 打印合并操作的信息
        print(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        # 将合并操作记录下来
        merges[pair] = idx
    print("tokens length:", len(tokens))
    print("ids length:", len(ids))
    print(f"compression ratio: {len(tokens) / len(ids):.2f}X")
    return merges

def encode(text, merges):
    # given a string, return list of integers (the tokens)
    tokens = list(text.encode("utf-8"))
    ids = tokens
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break # nothing else can be merged
        idx = merges[pair]
        ids = merge(ids, pair, idx)
    print("tokens length:", len(tokens))
    print("ids length:", len(ids))
    print(f"compression ratio: {len(tokens) / len(ids):.2f}X")
    return tokens

def decode(ids, vocab):
    # 将ids转换为字节串
    bytes = b''.join(vocab[idx] for idx in ids)
    text = bytes.decode("utf-8", errors="replace")

    return text

if __name__ == "__main__":
    # 读取语料库
    tokens = build_token("week14/corpus.txt")
    # 设置token大小
    vocab_size = 276
    # 合并次数
    num_merges = vocab_size - 256
    # 原始id
    ids = list(tokens)
    # 合并操作
    merges = build_merges(ids, num_merges)
    # 创建字表
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    # encode
    text = "你好，世界！"
    print(text)
    ids = encode(text, merges)
    print(ids)
    # decode
    text = decode(ids, vocab)
    print(text)
