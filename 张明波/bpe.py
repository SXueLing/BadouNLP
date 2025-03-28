/
---

## 核心代码实现 (tokenizer.py)

```python
import re
from collections import defaultdict
from heapq import heappush, heappop
from typing import Dict, List, Tuple

class BPETokenizer:
    def __init__(self, vocab_size: int = 500):
        """
        BPE分词器初始化
        :param vocab_size: 目标词表大小 (需大于256)
        """
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {}
    
    def train(self, corpus: str, verbose: bool = False) -> None:
        """
        训练BPE词表
        :param corpus: 训练语料文本
        :param verbose: 是否打印训练过程
        """
        # 初始化基础词表
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        # 转换为字节序列
        tokens = list(corpus.encode("utf-8"))
        ids = [int(b) for b in tokens]
        
        # 优先队列加速高频对查找
        heap = []
        pair_counts = defaultdict(int)
        
        # 初始统计
        for i in range(len(ids)-1):
            pair = (ids[i], ids[i+1])
            pair_counts[pair] += 1
        
        # 初始化堆
        for pair, count in pair_counts.items():
            heappush(heap, (-count, pair))  # 使用负数实现最大堆
            
        # 执行合并
        for idx in range(256, self.vocab_size):
            if not heap:
                break
                
            # 取最高频对
            count, pair = heappop(heap)
            count = -count  # 还原实际计数
            
            # 记录合并规则
            self.merges[pair] = idx
            
            # 生成新token
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
            if verbose:
                try:
                    print(f"Merged {pair} -> {idx} ({self.vocab[idx].decode('utf-8')})")
                except UnicodeDecodeError:
                    print(f"Merged {pair} -> {idx} (binary)")
            
            # 更新统计信息
            new_counts = defaultdict(int)
            i = 0
            while i < len(ids):
                if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                    new_id = idx
                    if i > 0:
                        left_pair = (ids[i-1], new_id)
                        new_counts[left_pair] += 1
                    if i < len(ids)-2:
                        right_pair = (new_id, ids[i+2])
                        new_counts[right_pair] += 1
                    i += 2
                else:
                    if i > 0:
                        left_pair = (ids[i-1], ids[i])
                        new_counts[left_pair] += 1
                    i += 1
            
            # 更新堆
            for p, c in new_counts.items():
                heappush(heap, (-c, p))
                
            ids = self._apply_merge(ids, pair, idx)
    
    def _apply_merge(self, ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        """应用合并规则到序列"""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def encode(self, text: str) -> List[int]:
        """编码文本为token序列"""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            # 查找可合并的最优对
            best_pair = None
            best_priority = float("inf")
            for i in range(len(ids)-1):
                pair = (ids[i], ids[i+1])
                if pair in self.merges:
                    if self.merges[pair] < best_priority:
                        best_priority = self.merges[pair]
                        best_pair = pair
            
            if not best_pair:
                break
            
            # 应用合并
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids)-1 and (ids[i], ids[i+1]) == best_pair:
                    new_ids.append(self.merges[best_pair])
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """解码token序列为文本"""
        bytes_data = b"".join(self.vocab[idx] for idx in ids)
        return bytes_data.decode("utf-8", errors="replace")

if __name__ == "__main__":
    tokenizer = BPETokenizer(vocab_size=300)
    
    # 训练语料
    corpus = "自然语言处理是人工智能的重要方向。"
    tokenizer.train(corpus, verbose=True)
    text = "自然语言"
    encoded = tokenizer.encode(text)
    print(f"编码结果: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"解码结果: {decoded}")
