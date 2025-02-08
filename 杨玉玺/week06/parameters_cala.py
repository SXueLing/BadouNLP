import torch
import math
import numpy as np
from transformers import BertModel

"""
bert每层的参数
"""

model = BertModel.from_pretrained(r'F:\NLP\bert-base-chinese',return_dict=False)
n = 2 # 输入最大句子的个数
vocab = 21128  # 词表数目
max_sequence_length = 512  # 最大句子长度
embedding_size = 768   # embedding 的维度
hide_size = 3072   # 隐藏层维度
num_layers = 12   # 隐藏层层数

# embedding过程中的参数
# todo token emdeddings = vocab * embedding_size
# todo segment embeddings = n * embedding_size
# todo position embeddings = max_sequence_length * embedding_size
# todo layer_norm层参数 = embedding_size + embedding_size
embedding_parameters =vocab * embedding_size  + n * embedding_size  + max_sequence_length * embedding_size  +embedding_size + embedding_size

# self_attention 过程中的参数
# todo QKV 分别求出权重w和偏置b
self_attention_parameters= (embedding_size * embedding_size + embedding_size)*3

# self_attention_out 的参数
# todo embedding_size * embedding_size layer_norm层参数 作用是防止梯度消失或者梯度爆炸
self_attention_out_parameters = embedding_size * embedding_size + embedding_size + embedding_size + embedding_size

# feed forward 参数 todo 两个线性层加一个 layer_norm层参数
# todo 第一个线性层 = embedding_size * hide_size + hide_size
# todo 第二个线性层 = hide_size * embedding_size + embedding_size
# todo  layer_norm层参数 = embedding_size + embedding_size
feed_forward_parameters =embedding_size * hide_size + hide_size + hide_size * embedding_size + embedding_size + embedding_size + embedding_size

# pool 层参数
pool_parameters = embedding_size * embedding_size + embedding_size


# 模型总参数
all_parameters = embedding_parameters+ (self_attention_parameters  + self_attention_out_parameters + feed_forward_parameters)*num_layers + pool_parameters

# 打印出模型的实际参数
print("模型实际参数个数为%d" % sum(p.numel() for p in model.parameters()))
# 手动计算模型的所有参数
print("diy计算参数个数为%d" % all_parameters)
