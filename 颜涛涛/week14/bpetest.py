import os


def bpeData(folder_path):
    # 定义所有的内容组合的空间。
    pre_bpe = []
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
    elif not os.path.isdir(folder_path):
        print(f"'{folder_path}' is not a directory.")
    else:
        # 使用os.walk遍历所有文件
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                print(full_path)
                with open(full_path, encoding="utf-8") as f:
                    lines = f.readlines()
                    processed_lines = []
                    for line in lines:
                        # 跳过空白行
                        if not line.strip():
                            continue
                        tokens = line.encode("utf-8")
                        tokens = list(map(int, tokens))
                        print('---')
                        print(line)
                        print("length:", len(line))
                        print('---')
                        print(tokens)
                        print("length:", len(tokens))
                        pre_bpe.extend(tokens)
        print(pre_bpe)
        print(len(pre_bpe))
    return pre_bpe


def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):  # Pythonic way to iterate consecutive elements
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


def bpe_func(vocab_size, ids):
    # vocab_size = 276  # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
    num_merges = vocab_size - 256
    # ids = list(tokens)  # copy so we don't destroy the original list

    merges = {}  # (int, int) -> int
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        print(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    return merges

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
      break # nothing else can be merged
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
  return tokens


def decode(ids):
  # given ids (list of integers), return Python string
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text


if __name__ == '__main__':
    # 定义要遍历的文件夹路径
    folder_path = r'F:\NLP\资料\week14 对话系统\week14 大语言模型相关第四讲\RAG\dota2英雄介绍-byRAG\Heroes'
    vocab_size = 30000
    ids = bpeData(folder_path)
    merges = bpe_func(vocab_size, ids)
    # print('--------')
    # print(len(merges))
    # print('--------')
    # print(merges)

    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]

    print(encode("技能描述"))
    print(decode(encode("技能描述")))


