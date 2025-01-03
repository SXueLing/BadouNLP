# 计算Bert参数量


import numpy as np

# BERT-base 参数
vocab_size = 30522  # 词汇表大小
hidden_size = 768  # 隐藏层大小
num_hidden_layers = 12  # 隐藏层的数量
num_attention_heads = 12  # 注意力头的数量
intermediate_size = 3072  # 前馈网络的中间层大小
max_position_embeddings = 512  # 最大序列长度

# 计算词嵌入层参数
token_embeddings = vocab_size * hidden_size
position_embeddings = max_position_embeddings * hidden_size
segment_embeddings = 2 * hidden_size

# 计算多头注意力层参数
attention_head_size = hidden_size // num_attention_heads
attention_layer_params = (3 * hidden_size * attention_head_size) + (hidden_size * hidden_size)
attention_layer_norm = hidden_size * 2

# 计算前馈网络层参数
intermediate_layer_params = (hidden_size * intermediate_size) + (intermediate_size * hidden_size)
output_layer_params = intermediate_layer_params

# 计算总参数量
total_params = (token_embeddings + position_embeddings + segment_embeddings) + \
               (num_hidden_layers * (attention_layer_params + attention_layer_norm + output_layer_params))

print(f"BERT-base model has {total_params} parameters.")
