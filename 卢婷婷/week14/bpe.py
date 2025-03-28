from collections import defaultdict, Counter

def get_stats(vocab):
    """统计词表中相邻字符对的频率"""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    """合并词表中出现频率最高的字符对"""
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]
    return new_vocab

def build_bpe_vocab(text, num_merges):
    """构建BPE词表"""
    # 初始化词表，将文本分割为字符并统计频率
    vocab = defaultdict(int)
    for word in text.split():
        vocab[' '.join(word)] += 1

    # 进行num_merges次合并操作
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        # 找到频率最高的字符对
        best_pair = max(pairs, key=pairs.get)
        # 合并字符对并更新词表
        vocab = merge_vocab(best_pair, vocab)
        print(f"Merge {i + 1}: {best_pair} -> {''.join(best_pair)}")

    # 返回最终的BPE词表
    return vocab

# 示例
text = "The ancient buildings, the people in traditional costumes and the bustling market make one feel as if they were in a dream."
num_merges = 10
bpe_vocab = build_bpe_vocab(text, num_merges)

# 输出最终的BPE词表
print("\nFinal BPE Vocabulary:")
for word in bpe_vocab:
    print(word)
