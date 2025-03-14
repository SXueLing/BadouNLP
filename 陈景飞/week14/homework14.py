from collections import defaultdict, Counter

def get_stats(vocab):
    """统计相邻符号对的频率"""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    print("pairs===", pairs)
    return pairs

def merge_vocab(pair, vocab_in):
    """合并词表中的指定符号对"""
    vocab_out = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word in vocab_in:
        new_word = word.replace(bigram, replacement)
        vocab_out[new_word] = vocab_in[word]
    return vocab_out

def build_bpe_vocab(corpus, num_merges):
    """基于 BPE 构建词表"""
    # 初始化词表为字符级别
    vocab = defaultdict(int)
    for sentence in corpus:
        chars = " ".join(list(sentence))  # 将句子拆分为字符
        vocab[chars] += 1
    print("vocab=1==",vocab)

    # 逐步合并符号对
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break  # 如果没有可合并的对，提前退出
        best_pair = max(pairs, key=pairs.get)  # 选择频率最高的对
        vocab = merge_vocab(best_pair, vocab)
        print("vocab=2==", vocab)

    # 提取最终的词表
    bpe_vocab = set()
    for word in vocab:
        for token in word.split():
            bpe_vocab.add(token)
    return sorted(bpe_vocab)

# 示例使用
if __name__ == "__main__":
    # 示例语料
    corpus = [
        "DeepSeek 是一家专注于实现 AGI（通用人工智能）的中国公司，成立于 2023 年。其使命是让 AGI 成为现实，并致力于成为 AGI 时代最核心的基础设施提供者。DeepSeek 的核心团队由一批经验丰富的技术专家和研究人员组成，在人工智能领域拥有深厚的积累。"
    ]

    num_merges = 20  # 合并次数
    bpe_vocab = build_bpe_vocab(corpus, num_merges)

    print("构建的 BPE 词表：")
    for token in bpe_vocab:
        print(token)
