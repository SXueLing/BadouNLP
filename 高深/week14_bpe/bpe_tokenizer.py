from collections import Counter
import os
from typing import List, Dict, Tuple

class BPETokenizer:
    def __init__(self, train_text_dir: str) -> None:
        train_ids, vocab_size = self.prepare_train_data(train_text_dir)
        bpe_ids, merges, vocab = self.train_tokenizer(train_ids, vocab_size)
        self.merges = merges
        self.vocab = vocab

        print("-----BPE Tokenizer training results-----")
        print("original token length:", len(train_ids))
        print("encoded token length:", len(bpe_ids))
        print(f"compression ratio: {len(train_ids) / len(bpe_ids):.2f}X")
        print("----------------------------------------")
    
    def read_text_files(self,folder_path: str) -> str:
        text = ""
        for file in os.listdir(folder_path):
            with open(os.path.join(folder_path, file), "r") as f:
                text += f.read()
            text += "\n\n"
        return text
    
    def prepare_train_data(self, train_text_dir: str) -> Tuple[List[int], int]:
        # step 1: build data
        text = self.read_text_files(train_text_dir)
        vocab_size = len(set(text))
        print(f"unique characters in training text (vocab size): {vocab_size}")
        tokens = text.encode("utf-8") # raw bytes
        tokens = list(map(int, tokens)) # convert list of bytes to a list of integers in range 0..255 
        return tokens, vocab_size

    # step 2: count the frequency of each token
    def get_stats(self, ids: List[int]) -> Dict[Tuple[int, int], int]:
        return Counter(zip(ids, ids[1:]))

    # step 3: merge the tokens with the highest frequency
    def merge(self, ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
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


    def train_tokenizer(self,ids: List[int], vocab_size: int = 276) -> List[int]:
        # ---
        # vocab_size = 276 # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
        num_merges = vocab_size - 256
        print(f"training BPE with {num_merges} merges")

        merges = {} # (int, int) -> int. map of merges: key=(token i, token i+1), value=merged new token id
        progress = 0
        for i in range(num_merges):
            stats = self.get_stats(ids) # get 2-gram frequency counts
            pair = max(stats, key=stats.get) # find the most frequent pair
            idx = 256 + i
            # print(f"merging {pair} into a new token {idx}")
            ids = self.merge(ids, pair, idx) # merge pair into new token indexed by idx, in place
            merges[pair] = idx
            
            # log progress
            progress += 1
            if (progress % 10) == 0:
                print(f"progress: {(progress / num_merges * 100):.2f}%")

        # build vocab using merges map
        vocab = {idx: bytes([idx]) for idx in range(256)} # token index -> token byte
        for (p0, p1), idx in merges.items():
            vocab[idx] = vocab[p0] + vocab[p1] # add merged tokens to vocab
        return ids, merges, vocab


    def encode(self, text: str) -> List[int]:
        tokens = text.encode("utf-8")
        tokens = list(map(int, tokens))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing else can be merged
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens
        

    def decode(self, ids: List[int]) -> str:
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text


if __name__ == "__main__":
    # Build bpe tokenizer based on text data from /Heros/
    bpe_tokenizer = BPETokenizer("./Heroes/")

    # test encode and decode
    test_sentence = "技能描述：将一个敌人拖到风暴之灵所在位置的涡流。"
    encoded = bpe_tokenizer.encode(test_sentence)
    decoded = bpe_tokenizer.decode(encoded)
    print(f"Test Sentence '{test_sentence}'. Original UTF-8 encoding: {list(map(int, test_sentence.encode('utf-8')))}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    assert test_sentence == decoded