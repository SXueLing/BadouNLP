import re
from collections import Counter


class BPE:
    def __init__(self, num_merges=10):
        self.num_merges = num_merges
        self.merges = {}

    def train(self, corpus):

        vocab = Counter([" ".join(word) + " </w>" for word in corpus])

        for _ in range(self.num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)

            self.merges[best_pair] = best_pair[0] + best_pair[1]

            vocab = self._merge_vocab(best_pair, vocab)

    def encode(self, word):

        word = " ".join(word) + " </w>"
        tokens = word.split()

        while True:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            best_pair = min(pairs, key=lambda p: self.merges.get(p, float('inf')))

            if best_pair not in self.merges:
                break

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(self.merges[best_pair])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens[:-1]

    def _get_stats(self, vocab):

        pairs = Counter()
        for word, count in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += count
        return pairs

    def _merge_vocab(self, pair, vocab):
        pattern = re.escape(" ".join(pair))
        new_vocab = {}
        for word in vocab:
            new_word = re.sub(pattern, self.merges[pair], word)
            new_vocab[new_word] = vocab[word]
        return new_vocab
