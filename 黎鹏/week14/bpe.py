import re
import os
from collections import defaultdict


def get_stats(word_freq):
    pairs = defaultdict(int)
    for word, freq in word_freq.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_bigram(pair, word_freq):
    new_word_freq = {}
    bigram_str = ' '.join(pair)  # "a b" 格式
    new_bigram = ''.join(pair)  # "ab" 格式

    for word, freq in word_freq.items():
        new_word = re.sub(r'\b' + re.escape(bigram_str) + r'\b', new_bigram, word)
        new_word_freq[new_word] = freq

    return new_word_freq


def byte_pair_encoding(corpus, num_merges=10):
    word_freq = {" ".join(word) + " _": 1 for word in corpus}
    for _ in range(num_merges):
        stats = get_stats(word_freq)
        if not stats:
            break
        best_pair = max(stats, key=stats.get)  # 找到频率最高的 bigram
        word_freq = merge_bigram(best_pair, word_freq)
    subwords = set()
    for word in word_freq.keys():
        subwords.update(word.split())

    return subwords

corpus = ["low", "lower", "newest", "widest", "biggest", "big", "better", "tea", "teacher"]

subword_vocab = byte_pair_encoding(corpus, num_merges=10)

print("BPE 词表:", subword_vocab)
