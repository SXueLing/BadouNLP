import torch
import math
import torch.nn as nn
import numpy as np
from transformers import BertModel

model = BertModel.from_pretrained(r"..\bert-base-chinese", return_dict=False)
# 词表大小
vocab_len = 21128
# 句子数量
sent_len = 2
# 文本最大长度
max_len = 512
# embedding维度
embedding_size = 768
# 隐藏层维数
hide_size = 3072
# 隐藏层层数
num_layers = 12

# token_embedding 参数统计
token_embedding = vocab_len * embedding_size
# segment_embedding 参数统计
segment_embedding = sent_len * embedding_size
# position_embedding 参数统计
position_embedding = max_len * embedding_size
# layer_normalization 参数统计
layer_normalization = embedding_size + embedding_size
# embedding过程总参数统计
embedding_parameters = token_embedding + segment_embedding + position_embedding + layer_normalization

# self_attention过程的参数统计
self_attention_parameters = (embedding_size * embedding_size + embedding_size) * 3

# self_attention_out参数统计
self_attention_out_parameters = embedding_size * embedding_size + embedding_size + layer_normalization

# Feed Forward参数统计
feed_forward_parameters = embedding_size * hide_size + hide_size + embedding_size * hide_size + embedding_size + layer_normalization

# pool_fc层参数统计
pool_fc_parameters = embedding_size * embedding_size + embedding_size

# 总参数统计
all_parameters = embedding_parameters + (self_attention_parameters + self_attention_out_parameters + feed_forward_parameters) * num_layers + pool_fc_parameters
# 加载预训练模型参数求和
model_parameters = 0
for p in model.parameters():
    model_parameters += p.numel()

print(f"预训练模型实际参数个数：{model_parameters} 个人统计总参数个数：{all_parameters}")
