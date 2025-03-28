# 作业：实现BPE算法
import os
from collections import defaultdict,Counter
from typing import Dict,Tuple,List

class BPEDemo:
    def __init__(self,vocab_size = 500):
        self.vocab_size = vocab_size
        self.vocab: Dict[int,bytes] = {i:bytes([i]) for i in range(256)}
        self.merges: Dict[Tuple[int,int],int] = {}
    
    def _getstates(self,token_ids:List[int]) -> Counter:
        """使用Counter替代手动统计"""
        return Counter(zip(token_ids,token_ids[1:]))
    
    def _merge(self,token_ids: List[int],pair:Tuple[int, int],new_id: int)-> List[int]:
        """使用列表推导优化合并操作"""
        i,new_ids = 0,[]
        while i < len(token_ids):
            if i < len(token_ids) - 1 and (token_ids[i], token_ids[i+1]) == pair:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(token_ids[i])
                i += 1
        return new_ids
    
    def fir(self,corpus:str):
        """训练BPE词表（"""
        token_ids = list(corpus.encode("utf-8"))
        num_merges = self.vocab_size - 256

        for merge_step in range(num_merges):
            stats = self._getstates(token_ids)
            if not stats:
                break # 没有更多可合并的pair时提前终止

            # 获取最高频pair
            (most_common_pair, _), = stats.most_common(1)

            new_id = 256 + merge_step
            self.merges[most_common_pair] = new_id
            token_ids = self._merge(token_ids,most_common_pair,new_id)

            # 构建可逆词表
            self.vocab[new_id] = self.vocab[most_common_pair[0]] + self.vocab[most_common_pair[1]]

    def encode(self,text:str) -> List[int]:
        """使用栈结构加速合并"""
        tokens = list(text.encode("utf-8"))

        # 使用优先级缓存优化合并顺序
        while len(tokens) >= 2:
            pairs = list(zip(tokens,tokens[1:]))
            best_pair = min(
                pairs,
                key = lambda p:self.merges.get(p,float("inf")),
                default=None
            )
            if not best_pair or best_pair not in self.merges:
                break

            new_id = self.merges[best_pair]
            tokens = self._merge(tokens,best_pair,new_id)

        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """批量处理字节拼接"""
        byte_stream = b"".join(self.vocab[idx] for idx in token_ids)
        try:
            return byte_stream.decode("utf-8")
        except UnicodeDecodeError:
            return byte_stream.decode("uft-8",errors="replace")


if __name__ == "__main__":
    # 读取语料
    file_path = "week14/data/Heroes"
    corpus = []
    for filename in os.listdir(file_path):
        path = os.path.join(file_path,filename)
        with open(path,"r",encoding="utf-8") as f:
            corpus.append(f.read())
    full_corpus = "\n".join(corpus)
    #print(full_corpus[2])

    # 训练BPE
    bpe = BPEDemo(vocab_size=500)
    bpe.fir(full_corpus)

    # 测试编解码
    test_str = "齐天大圣"
    encoded = bpe.encode(test_str)
    decoded = bpe.decode(encoded)
    print(f"编码结果: {encoded}") #编码结果: [233, 189, 144, 276, 169, 379, 271, 163]
    print(f"解码结果: {decoded}") #解码结果: 齐天大圣
