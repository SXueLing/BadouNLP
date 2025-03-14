import os



def read_data(path):
    text =""
    for filename in os.listdir(path):
        # 即使已知是txt文件，仍建议校验扩展名
        if not filename.endswith(".txt"):
            continue  # 跳过非txt文件
        filepath = os.path.join(path, filename)

        # 异常处理（例如文件被占用或损坏）
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                content = content.replace("\n","").replace("\t","").replace("<br>","")
                text += content
        except Exception as e:
            print(f"读取 {filename} 失败: {str(e)}")

    return text


#按照bpe的思想，我们统计每个2元组出现次数

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

## 将ids中的pair对 替换为idx
def merge(ids, pair, idx):
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

def encode(text,merges):
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

def decode(ids,vocab):
  # given ids (list of integers), return Python string
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text


if __name__ == "__main__":
    text = read_data("data/Heroes")


    # text = text[:100]
    tokens = text.encode("utf-8")
    tokens = list(map(int, tokens))



    vocab_size = 306  # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
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

    print(merges)


    # text = encode(text,merges)
    # print(text)
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    # text_old = decode(text,vocab)

    print(text == decode(encode(text,merges),vocab))



