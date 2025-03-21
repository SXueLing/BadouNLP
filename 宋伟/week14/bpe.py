from collections import Counter
import re

def get_vocab(corpus):
    vocab = Counter()
    for word in corpus:
        vocab[" ".join(word) + " _"] += 1  # 添加终止符 _
    return vocab

def get_stats(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    new_vocab = {}
    bigram = re.escape(" ".join(pair))
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in vocab:
        new_word = pattern.sub("".join(pair), word)
        new_vocab[new_word] = vocab[word]
    return new_vocab

def bpe(corpus, num_merges=10):
    vocab = get_vocab(corpus)
    merges = []
    
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)
        merges.append(best_pair)
    
    return merges, vocab

# 示例语料
corpus = ["low", "lower", "newest", "widest"]
merges, final_vocab = bpe(corpus, num_merges=10)

print("BPE 合并操作:")
for i, merge in enumerate(merges):
    print(f"Step {i + 1}: {merge}")

print("\n最终词汇表:")
for token, freq in final_vocab.items():
    print(f"{token}: {freq}")
