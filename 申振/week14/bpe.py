import os
from collections import defaultdict, Counter


class BPETokenizer:
    def __init__(self, vocab_size=300):
        self.vocab = {}  # 词表：token_id -> bytes
        self.merges = {}  # 合并规则
        self.vocab_size = vocab_size

    def _preprocess(self, text):
        """文本预处理：按Unicode字符拆分"""
        return list(text.encode("utf-8"))

    def _get_stats(self, ids):
        """统计相邻字节对频率"""
        pairs = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            pairs[pair] += 1
        return pairs

    def _merge(self, ids, pair, idx):
        """执行合并操作"""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, corpus, verbose=False):
        """训练BPE词表"""
        # 初始词表（0-255为原始字节）
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        # 预处理文本
        ids_list = [self._preprocess(text) for text in corpus]

        # BPE迭代合并
        for i in range(256, self.vocab_size):
            # 统计所有文本中的字节对
            stats = defaultdict(int)
            for ids in ids_list:
                pairs = self._get_stats(ids)
                for k, v in pairs.items():
                    stats[k] += v

            if not stats:
                break  # 无更多可合并对

            # 选择最高频对
            best_pair = max(stats, key=stats.get)
            self.merges[best_pair] = i
            self.vocab[i] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # 更新所有文本
            ids_list = [self._merge(ids, best_pair, i) for ids in ids_list]

            if verbose:
                print(f"Merge {i - 255}/{self.vocab_size - 256}: {best_pair} -> {i}")

    def encode(self, text):
        """编码文本为token IDs"""
        ids = self._preprocess(text)
        while len(ids) >= 2:
            # 找最可能合并的对（优先合并训练过的对）
            pairs = self._get_stats(ids)
            pair_to_merge = None
            for pair in sorted(pairs, key=lambda x: -self.merges.get(x, -1)):
                if pair in self.merges:
                    pair_to_merge = pair
                    break

            if not pair_to_merge:
                break  # 无更多可合并对

            # 执行合并
            idx = self.merges[pair_to_merge]
            ids = self._merge(ids, pair_to_merge, idx)

        return ids

    def decode(self, ids):
        """解码token IDs为文本"""
        bytes_data = b""
        for idx in ids:
            bytes_data += self.vocab[idx]
        return bytes_data.decode("utf-8", errors="replace")

    def save_vocab(self, path):
        """保存词表"""
        import json
        with open(path, "w") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
            }, f)

    def load_vocab(self, path):
        """加载词表"""
        import json
        with open(path) as f:
            data = json.load(f)
            self.vocab_size = data["vocab_size"]
            self.merges = {tuple(map(int, k.split(","))): v for k, v in data["merges"].items()}
            # 重建vocab
            self.vocab = {idx: bytes([idx]) for idx in range(256)}
            for pair, idx in self.merges.items():
                self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

def load_corpus(dir_path = "./Heroes"):
    corpus = []
    for root, _, file in os.walk(dir_path):
        for filename in file:
            if filename.endswith("txt"):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding="utf-8") as infile:
                    for line in infile:
                        corpus.append(line.strip())
    return corpus

# ============================ 使用示例 ============================#
if __name__ == "__main__":
    # 示例文本库（替换为你的实际文本）
    corpus = load_corpus()
    #print(corpus)
    # 初始化并训练分词器
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(corpus, verbose=True)




    #tokenizer.load_vocab("bpe_vocab.json")
    # 编码测试
    text = "变形金刚"
    encoded_ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(encoded_ids)

    print("\n编码结果:")
    print("原始文本:", text)
    print("Token IDs:", encoded_ids)
    print("解码文本:", decoded_text)
    print("解码文本:", tokenizer.decode([267,152]))

    # 保存/加载词表
    tokenizer.save_vocab("bpe_vocab.json")
