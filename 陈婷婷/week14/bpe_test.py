
import json

class BPE():
    def __init__(self, config):
        self.config = config 
        self.vocab_size = config["vocab_size"]
        self.num_merges = self.vocab_size - 256
        self.newids = []

        
    
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

    def bpe_merge(self, ids, merges):
        for i in range(self.num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f"merging {pair} into a new token {idx}")
            ids = self.merge(ids, pair, idx)
            merges[pair] = idx 
        return merges, ids
    
    def encode(self, text, merges):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >=2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break
            idx = merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens
    
    def decode(self, ids, merges):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        tokens = b"".join(vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text 




class DataLoad():
    def __init__(self, config):
        self.config = config
        self.path = config["data_path"]
    
    def load(self):
        self.txt = ""
        with open(self.path, encoding="utf8") as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                content = line["content"]
                self.txt += content
        return self.txt


class EncodeDecode():
    def __init__(self, config):
        self.config = config


config = {
    "vocab_size": 5,
    "data_path": "/Users/jessicachan/nlp20/transformers-生成文章标题/sample_data.json"
}

loader = DataLoad(config)
bpe = BPE(config)
text = loader.load()
print(len(text))
merges = {}
ids = bpe.encode(text, merges)
print(len(ids))
merges, result = bpe.bpe_merge(ids, merges)
print(len(result))


test = "就俄罗斯免费医疗话题"
print(len(test))
print(test)
merges = {}
ids = bpe.encode(test, merges)
print(ids)
merges, result = bpe.bpe_merge(ids, merges)
print(result)
result_decode = bpe.decode(result, merges)
print(result_decode)
