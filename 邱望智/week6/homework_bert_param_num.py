# bert parameters
# Bert参数量 = embedding层参数 + 层数 * 每层的参数个数 + pooler参数
# 每层的参数个数 = MultiHeadAttention层参数 + MLP层参数

vocab = 21128
hidden_size = 768
num_layers = 12
max_sequence_length = 512
type_vocab_size = 2
h = hidden_size

# token embeddings
token_embeddings = vocab * h
# segment embeddings
segment_embeddings = type_vocab_size * h
# position embeddings
position_embeddings = max_sequence_length * h
# BERT的embedding由词向量（Token Embeddings）、块向量（Segment Embeddings）、位置向量（Position Embeddings）组成
# 外加embedding layer_norm层weight和bias
embedding_params = token_embeddings + segment_embeddings + position_embeddings + h + h

# MultiHeadAttention共有Q, K, V, O四个矩阵，每个矩阵的维度为(h, h)，外加bias
# 加上layer_norm层weight和bias
attention = 4 * (h * h + h) + h + h

# MLP共有两个全连接层，维度变化为h -> 4h -> h，外加上layer_norm层weight和bias
# MLP层参数量为：(h * 4h + 4h) + (4h * h + h) + h + h
mlp = (h * 4 * h + 4 * h) + (4 * h * h + h) + h + h

# pooler层
pooler = h * h + h

# all parameters
all_parameters = embedding_params + num_layers * (attention + mlp) + pooler
print(all_parameters)
