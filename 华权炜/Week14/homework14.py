import collections
import re

def get_pair_with_frequency(data):
    pairs = collections.defaultdict(int)
    for word, freq in data.items():
        sub_words = word.split()
        for i in range(len(sub_words)-1):
            pair = (sub_words[i], sub_words[i+1])
            pairs[pair] += freq
    return pairs

def merge_data_with_pair(pair, data):
    result = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in data:
        merged_word = p.sub(''.join(pair), word)
        result[merged_word] = data[word]
    return result

def build_vocab(train_data, num_merges):
    subwords = get_subwords(train_data)
    bpe_vocab = set(subwords.keys())
    for _ in range(num_merges):
        pairs = get_pair_with_frequency(train_data)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        if pairs[best_pair] == 1:
            break
        train_data = merge_data_with_pair(best_pair, train_data)
        merged_word = "".join(best_pair)
        bpe_vocab.add(merged_word)
        subwords = get_subwords(train_data)
        if best_pair[0] not in subwords:
            bpe_vocab.remove(best_pair[0])
        if best_pair[1] not in subwords:
            bpe_vocab.remove(best_pair[1])
    return bpe_vocab
