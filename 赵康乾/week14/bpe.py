import torch
import torch.nn as nn
import os
import glob

class BPE:
    def __init__(self, vocab_size = 280, corpus_path = 'data/Heroes'):
        self.vocab_size = vocab_size
        self.corpus = self.load_corpus(corpus_path)
        self.merges = {}
        self.raw_ids = self.raw_ids(self.corpus)
        self.final_ids = self.final_ids(self.raw_ids)
        self.vocab = self.build_vocab()

    def load_corpus(self, corpus_path):
        current_path = os.getcwd()
        print(f'当前工作路径: {current_path}')
        corpus = ''
        # 读取所有的txt，去除空白行, 连续储存
        corpus_paths = glob.glob(os.path.join(corpus_path, '*.txt'))
        for path in corpus_paths:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        corpus += line
        return corpus

    # 把字符串语料用utf-8编码成初始id
    def raw_ids(self, corpus):
        raw_ids = list(map(int, corpus.encode('utf-8')))
        return raw_ids
    
    # 统计id pair的频次
    def get_stats(self, ids):
        pair_counts = {}
        for pair in zip(ids,ids[1:]):
            pair_counts[pair] = pair_counts.get(pair,0) + 1
        return pair_counts
    
    def merge_pairs(self, ids, pair_counts, new_idx):
        new_ids = []
        # 找到频次最高的pair
        max_pair = max(pair_counts, key=pair_counts.get) # key = dict.get按照value排序并返回键
        self.merges[max_pair] = new_idx # 记录merge的pair和新的id，用于解码
        print(max_pair, pair_counts[max_pair])

        # 遍历ids，将pair替换成新的id
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == max_pair[0] and ids[i+1] == max_pair[1]:
                new_ids.append(new_idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def final_ids(self, ids):
        merge_times = self.vocab_size - 255
        for i in range(1,merge_times+1):
            pair_counts = self.get_stats(ids)
            new_idx = 255 +i
            ids = self.merge_pairs(ids, pair_counts, new_idx)
        return ids
    
    def build_vocab(self):
        vocab = {idx:bytes([idx]) for idx in range(256)}
        for (p0,p1),idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        # print(vocab)
        return vocab
    
    def decode(self,ids):
        # given ids (list of integers), return Python string
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8",errors="replace")
        return text
    
    def encode(self, text):
        # given a string, return list of integers (the tokens)
        tokens = list(text.encode("utf-8"))
        print(self.merges)
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing else can be merged
            idx = self.merges[pair]
            tokens = self.merge_pairs(tokens, pair, idx)
        return tokens

if __name__ == "__main__":
    bpe = BPE(260)
    print(len(bpe.raw_ids))
    print(len(bpe.final_ids))
    tokens = bpe.encode("英雄矮人直升机的技能是高射火炮")
    print(tokens)
    print(bpe.decode(tokens))
