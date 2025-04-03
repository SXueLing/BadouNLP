import re

"""
基于bpe的文本压缩的原理，实现NLP中token的编码
"""


class BPE(object):
    def __init__(self, corpus_path, vocab_szie=1000):
        self.corpus_path = corpus_path # 超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
        self.vocab_szie = vocab_szie - 256
        self.trans_corpus()

    def trans_corpus(self):
        """
        将文本编码为utf-8
        """
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
        text = re.sub(r'<.*?>', '', text) #清洗数据，去掉文本中的标签符号
        text = re.sub(r'[ \t]+', '', text) #清洗数据，去掉制表符
        tokens = text.encode("utf-8")
        self.tokens_original = self.tokens = list(map(int, tokens))

    def run(self):
        """
        bpe编码过程
        1 先统计相邻unicode重复的次数
        2 找到重复次数最多的两两组合
        3 unicode中文最小单位为一个字节，所以两两组合后的新下标值从256开始
        4 根据2，3的结果构建重新编码后的集合
        """
        merges = {}  # (int, int) -> int
        for i in range(self.vocab_szie):
            stats = self.get_stats()
            pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f"merging {pair} into a new token {idx}")
            self.tokens = self.merge(pair, idx)
            merges[pair] = idx
        # 构建完整的字表
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (uni0, uni1), idx in merges.items():
            vocab[idx] = vocab[uni0] + vocab[uni1] # 将压缩后的unicode对组合后加入字表
            # print(idx, vocab[idx])
        return merges, vocab

    def get_stats(self):
        counts = {}
        for pair in zip(self.tokens, self.tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, pair, idx):
        newids = []
        i = 0
        while i < len(self.tokens):
            if i < len(self.tokens) - 1 and self.tokens[i] == pair[0] and self.tokens[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(self.tokens[i])
                i += 1
        return newids


def main():

    path = r"./哈斯卡.txt"
    bpe = BPE(path, 300)
    merges, vocab = bpe.run()

    print(merges)
    print(vocab)


if __name__ == "__main__":
    main()
