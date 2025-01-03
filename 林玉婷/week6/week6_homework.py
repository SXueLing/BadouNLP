import torch
import math
import numpy as np
from transformers import BertModel



bert = BertModel.from_pretrained(r"/Users/lindsay/Desktop/人工智能/bert-base-chinese", return_dict=False)
n = 2                       # 输入最大句子个数
vocab = 21128               # 词表数目
max_sequence_length = 512   # 最大句子长度
embedding_size = 768        # embedding维度
hide_size = 3072            # 隐藏层维数
num_layers = 12              # 隐藏层层数

# 首先是embedding层，分三个，token embeddings，segment embeddings，position embeddings
token_size = vocab * embedding_size + n*embedding_size + max_sequence_length * embedding_size
# 然后是self attention层
q = embedding_size * embedding_size + embedding_size
k = embedding_size * embedding_size + embedding_size
v = embedding_size * embedding_size + embedding_size
self_attention_size = q + k + v
# self_attention 还有输出参数,也要过线性层 X+Z
self_attention_out = embedding_size * embedding_size + embedding_size + embedding_size + embedding_size
# feed forward层 有两个线性层加一个激活层,第一个线性层输入是L*768，输出是L*3072
feed_forward_size = hide_size * embedding_size +  hide_size + embedding_size * hide_size + embedding_size
+ (embedding_size + embedding_size) # x+z
# 池化层
pool_size = embedding_size * embedding_size +embedding_size

#总参数
total_params = (token_size + (self_attention_size + self_attention_out +feed_forward_size) * num_layers
                + pool_size +pool_size)

print("模型实际参数个数为%d" % sum(p.numel() for p in bert.parameters()))
print("diy计算参数个数为%d" % total_params)
