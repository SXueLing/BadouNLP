import os

corpus_dir = r"D:\BaiduNetdiskDownload\八斗NLP课程\week14 大模型相关内容第四讲\第十四周 大模型相关内容第四讲\week14 大语言模型相关第四讲\RAG\dota2英雄介绍-byRAG\Heroes"
path = os.listdir(corpus_dir)
# 构建语料库
# with open("all_text.txt", "w", encoding="utf8") as f:
#     for text in path:
#         if text.endswith(".txt"):
#             with open(os.path.join(corpus_dir, text), encoding="utf8") as k:
#                 f.write(k.read())

with open("all_text.txt", encoding="utf8") as f:
    tokens = f.read()
tokens = tokens.encode("utf8")
tokens = list(map(int, tokens))
# print(tokens)

ids = tokens

# 得到pair频率
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    # print(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    return counts

# print(get_stats(tokens))

def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def encode(ids):
    vocab_size = 276
    merge_nums = vocab_size - 256
    merge_dict = {}
    for i in range(merge_nums):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        ids = merge(ids, pair, idx)
        merge_dict[idx] = pair

    return ids, merge_dict

ids, merge_dict = encode(ids)
# print(ids)
# print(len(tokens))
# print(len(ids))

vocab = {idx: bytes([idx]) for idx in range(256)}
for idx, (p0, p1) in merge_dict.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(sentence):
    sentence = b"".join([vocab[idx] for idx in sentence])
    text = sentence.decode("utf8", errors="replace")
    return text

sentence = [275, 49, 270, 229, 134, 165, 231, 129, 171, 231, 136, 134, 229, 135, 187, 44, 32, 275, 230, 143, 143, 266,
            176, 270, 229, 134, 165, 233, 173, 130, 229, 164, 167, 229, 184, 157, 229, 144, 145, 257, 128, 257, 170,
            230, 149, 140, 230, 150, 185, 229, 141, 149, 228, 189, 141, 267, 145, 229, 176, 132, 229, 134, 165, 233,
            173,130, 231, 129, 171, 231, 144, 131, 260, 233, 128, 160, 230, 136, 144, 229, 141, 179, 230, 151, 182,
            228,188, 164, 274, 179, 229, 146, 140, 231, 156, 169, 230, 153, 149, 260, 229, 185, 182, 230, 140, 129,
            231,187, 173, 229, 175, 185, 231, 155, 174, 230, 160, 135, 233, 128, 160, 230, 136, 144, 228, 188, 164, 274,
            179, 229, 146, 140, 229, 135, 143, 233, 128, 159, 262, 231, 155, 174, 230, 160, 135, 233, 153, 132, 266,
            145, 267, 172, 229, 148, 164, 259, 233, 170, 183, 233, 171, 133, 229, 133, 181, 228, 188, 154, 271, 168,
            232, 180, 159, 233, 157, 162, 230, 149, 136, 230, 158, 156, 259, 230, 140, 129, 231, 187, 173, 230, 151,
            182, 233, 151, 180, 229, 134, 133, 232, 142, 183, 229, 190, 151, 231, 167, 187, 229, 138, 168, 233, 128,
            159, 229, 186, 166, 229, 146, 140, 230, 148, 187, 229, 135, 187, 233, 128, 159, 229, 186, 166, 229, 138,
            160, 230, 136, 144, 262, 10]

print(decode(sentence))
