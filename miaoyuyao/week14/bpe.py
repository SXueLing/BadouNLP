import os
from collections import defaultdict, Counter
import re


def preprocess(text):
    words = re.findall(r"\w+", text.lower())
    return [" ".join(list(word)) + " </w>" for word in words]


def get_stats(words):
    pairs = defaultdict(int)
    for word, freq in words.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_vocab(pair, words):
    merged_word = "".join(pair)
    new_words = {}
    for word in words:
        new_word = word.replace(" ".join(pair), merged_word)
        new_words[new_word] = words[word]
    return new_words


class BPETokenizer:
    def __init__(self):
        self.vocab = set()
        self.id_to_token = None
        self.token_to_id = None
        self.merges = {}
        self.word_freq = None

    def train(self, text, num_merges):
        words = preprocess(text)
        self.word_freq = Counter(words)
        for word in self.word_freq:
            for symbol in word.split():
                self.vocab.add(symbol)
        for i in range(num_merges):
            pairs = get_stats(self.word_freq)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.word_freq = merge_vocab(best_pair, self.word_freq)
            self.merges[best_pair] = i
            merged_word = "".join(best_pair)
            self.vocab.add(merged_word)
        self.token_to_id = {token: idx for idx, token in enumerate(sorted(self.vocab))}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def tokenize(self, text):
        words = preprocess(text)
        tokens = []
        for word in words:
            symbols = word.split()
            while len(symbols) > 1:
                pairs = get_stats({" ".join(symbols): 1})
                if not pairs:
                    break
                best_pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
                if best_pair not in self.merges:
                    break
                merged_word = "".join(best_pair)
                symbols = " ".join(symbols).replace(" ".join(best_pair), merged_word).split()
            tokens.extend(symbols)
        return tokens

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.token_to_id[token] for token in tokens if token in self.token_to_id]

    def decode(self, ids):
        tokens = [self.id_to_token[idx] for idx in ids]
        return "".join(tokens).replace("</w>", " ").strip()


# 示例用法
if __name__ == "__main__":
    dir_path = r"D:\BaiduNetdiskDownload\AI\第十四周 大模型相关内容第四讲\week14 大语言模型相关第四讲\week14 大语言模型相关第四讲\RAG\dota2英雄介绍-byRAG\Heroes"
    corpus = ""
    for path in os.listdir(dir_path):
        path = os.path.join(dir_path, path)
        with open(path, encoding="utf8") as f:
            text = f.read()
            corpus += text + '\n'
    bpe_tokenizer = BPETokenizer()
    bpe_tokenizer.train(corpus, num_merges=10)
    print("词表:", bpe_tokenizer.vocab)
    encoded_ids = bpe_tokenizer.encode("魅惑魔女")
    print("编码结果:", encoded_ids)
    decoded_text = bpe_tokenizer.decode(encoded_ids)
    print("解码结果:", decoded_text)
