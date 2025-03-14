# 基于一篇文章，使用BPE算法进行分词

from collections import defaultdict
import os

def read_corpus(folder_path):
    corpus = ""
    for file_name in os.listdir(folder_path):
        with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                corpus += line + " "
    return [list(corpus.encode('utf-8'))]

def get_stats(sequences):
    pairs = defaultdict(int)
    for seq in sequences:
        for i in range(len(seq)-1):
            pairs[(seq[i], seq[i+1])] += 1
    return pairs

def merge_sequences(sequences, best_pair):
    merged_sequences = []
    a, b = best_pair
    for seq in sequences:
        i = 0
        new_seq = []
        while i < len(seq):
            if i < len(seq)-1 and seq[i] == a and seq[i+1] == b:
                new_seq.append(best_pair)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        merged_sequences.append(new_seq)
    return merged_sequences

def train_bpe(sequences, num_merges=10):
    merges = {}
    for _ in range(num_merges):
        pairs = get_stats(sequences)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        merged_id = len(merges) + 256 
        merges[best_pair] = merged_id
        sequences = merge_sequences(sequences, best_pair)
        for key, value in merges.items():
            l_key = list(flatten_tuple(key))
            bytes_word = bytes(l_key).decode('utf-8', errors='replace')
            print(f"第{value}个新词: {bytes_word} ({key})")
    return merges

def encode(text, merges):
    # 转换为 UTF-8 字节序列
    byte_seq = list(text.encode('utf-8'))
    # 应用合并规则
    for (a, b), merged_id in merges.items():
        i = 0
        while i < len(byte_seq)-1:
            if byte_seq[i] == a and byte_seq[i+1] == b:
                byte_seq = byte_seq[:i] + [merged_id] + byte_seq[i+2:]
            else:
                i += 1
    return byte_seq

def decode(encoded_seq, merges):
    # 反向查找合并规则
    reverse_merges = {v: k for k, v in merges.items()}
    # 逐步拆分合并符号
    while True:
        changed = False
        for i in range(len(encoded_seq)):
            if encoded_seq[i] in reverse_merges:
                a, b = reverse_merges[encoded_seq[i]]
                encoded_seq = encoded_seq[:i] + [a, b] + encoded_seq[i+1:]
                changed = True
                break
        if not changed:
            break
    # 转换为字节并解码为文本
    print("解码后的序列为:", encoded_seq)
    byte_seq = bytes(encoded_seq)
    return byte_seq.decode('utf-8', errors='replace')

def flatten_tuple(nested_tuple):
    flattened = []
    for item in nested_tuple:
        if isinstance(item, tuple):
            flattened.extend(flatten_tuple(item))
        else:
            flattened.append(item)
    return tuple(flattened)

# 读取训练数据并训练
sequences = read_corpus('../dota2英雄介绍-byRAG/Heroes')
merges = train_bpe(sequences, num_merges=50)
print("合并规则（字节对 → 新符号ID）:", merges)

# 测试编码解码
text = "在毕生为战争、起义、暴动和革命服务后，军中名人奥雷尔终于感到厌倦了。"
encoded = encode(text, merges)
decoded = decode(encoded, merges)

print("原始文本:", text)
print("编码序列:", encoded)
print("解码结果:", decoded)
