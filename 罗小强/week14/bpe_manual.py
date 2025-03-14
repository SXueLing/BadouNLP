# -*- coding: utf-8 -*-
"""
bpe_manual.py.py
描述: 
作者: TomLuo 21429503@qq.com
日期: 3/12/2025
版本: 1.0
"""
import json

class BPE:
    def __init__(self, vocab_size, num_merges):
        self.vocab_size = vocab_size
        self.num_merges = num_merges
        self.merges = {}

    def train(self, tokens):
        vocab = {chr(i): i for i in range(256)}  # 初始化ASCII字符集
        ids = list(tokens)  # 复制原始列表
        for i in range(self.num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f"merging {pair} into a new token {idx}")
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx
        self.vocab = {idx: ''.join(map(str, pair)) for pair, idx in self.merges.items()}
        self.vocab.update({i: chr(i) for i in range(256)})  # 添加ASCII字符

    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        for pair, idx in self.merges.items():
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def decode(self, tokens):
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        decoded_text = bytearray()
        for token in tokens:
            if token < 256:
                decoded_text.append(token)
            else:
                pair = reverse_vocab[token]
                decoded_text.extend(map(int, pair.split(',')))
        return decoded_text.decode("utf-8")
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4)
        print(f"词表已保存到 {path} 文件中")

def read_content():
    data_path = r'D:\AI\BadouNLP\罗小强\week11\news.json'
    # 读取输入文本
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
        text = ''
        for title, content in data:
            text += title + content
    return text
text = read_content()
# 训练BPE模型
bpe = BPE(vocab_size=5000, num_merges=20)
bpe.train(tokens=list(text))
bpe.save('encoder.json')

#随机生成中文，英文混合文本
test_text = "안녕하세요 👋 (hello in Korean!)"
# 使用训练好的BPE模型进行编码和解码
encoded_tokens = bpe.encode(test_text)
print("编码后的tokens:", encoded_tokens)
decoded_text = bpe.decode(encoded_tokens)
print("解码后的文本:", decoded_text)
# 验证编码和解码是否一致
assert test_text == decoded_text, "编码和解码不一致"
print("编码和解码一致，测试通过")
