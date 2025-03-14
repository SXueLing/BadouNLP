import os

class BPE:
    def __init__(self, filename, vocab_size):
        self.vocab_size = vocab_size
        self.merges = self.train_bpe(filename, vocab_size)
        self.vocab = self.build_vocab()
    
    def train_bpe(self, filename, vocab_size):
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()

        tokens = text.encode("utf-8") # raw bytes
        tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience

        num_merges = vocab_size - 256
        ids = list(tokens)

        merges = {}
        for i in range(num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f"merging {pair} into a new token {idx}")
            ids = self.merge(ids, pair, idx)
            merges[pair] = idx

        return merges

    def build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def decode(self, ids):
        # given ids (list of integers), return Python string
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # given a string, return list of integers (the tokens)
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing else can be merged
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def save_vocab(self, vocab_file):
        """将词汇表保存到文件中"""
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for idx in sorted(self.vocab.keys()):
                token = self.vocab[idx]
                f.write(f"{idx}\t{token.decode('utf-8', errors='replace')}\n")

def merge_txt_files(directory, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 遍历目录中的所有文件
        for filename in os.listdir(directory):
            # 构建文件的完整路径
            file_path = os.path.join(directory, filename)
            # 打开并读取txt文件内容
            with open(file_path, 'r', encoding='utf-8') as infile:
                # 将读取的内容写入输出文件
                outfile.write(infile.read())
                outfile.write('\n')

directory = 'D:/NLP/week14/RAG/Heroes'  # txt文件所在目录
corpus = 'corpus.txt'  # 输出文件名
merge_txt_files(directory, corpus)  #合并txt文件为语料

vocab_size = 1000
bpe = BPE(corpus , vocab_size)
text = "回音重踏, 技能描述：持续施法 - 上古巨神与他的灵体游魂一起践踏地面，对敌方单位造成伤害并使他们在原地昏迷。上古巨神造成物理伤害，灵体游魂造成魔法伤害。"
print(bpe.encode(text))
print(bpe.decode(bpe.encode(text)) == text)

vocab_file = 'vocab.txt'  # 保存词汇表的文件名
bpe.save_vocab(vocab_file)
