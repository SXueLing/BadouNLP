import os

from zhipuai import ZhipuAI


def call_large_model(content: str):
    with open('api_key.txt', "r", encoding="utf-8") as file:
        key = file.read()
    client = ZhipuAI(api_key=key)
    response = client.chat.completions.create(
        model="glm-4-flash",  # 免费模型
        messages=[
            {"role": "user", "content": content},
        ],
    )
    return response.choices[0].message.content


def get_utf8_code(text: str):
    return list(map(int, text.encode("utf-8")))


# 按照bpe的思想，我们统计每个2元组出现次数
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


def load_hero_data(folder_path):
    hero_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
                intro = file.read()
                hero = file_name.split(".")[0]
                hero_data[hero] = intro
    return hero_data


def decode(ids, vocab):
    # given ids (list of integers), return Python string
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text


def encode(text, merges):
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


def main():
    vocab_size = 360  # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
    num_merges = vocab_size - 256
    text = ''
    for t in load_hero_data('./RAG/dota2英雄介绍-byRAG/Heroes').values():
        text += t
    ids = list(get_utf8_code(text))

    merges = {}  # (int, int) -> int
    for i in range(num_merges):
        stats = get_stats(ids)
        # print(sorted(((v, k) for k, v in stats.items()), reverse=True))
        pair = max(stats, key=stats.get)
        idx = 256 + i
        # print(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    print(merges)
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    print(vocab)  # 最终的词表
    for i in range(256, 359):  # 打印看一下合并的词有哪些
        print(decode(vocab[i], vocab))


if __name__ == '__main__':
    main()
