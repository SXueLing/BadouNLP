def count_bert_params(vocabulary_size, embedding_dim, num_transformer_layers, max_position_embeddings, num_attention_heads=12):
    # 计算 embedding 层的参数
    # 1. Token Embedding（词嵌入层）参数
    token_embedding_params = vocabulary_size * embedding_dim  # 词汇表大小 * 嵌入维度
    
    # 2. Segment Embedding（分段嵌入层）参数
    segment_embedding_params = 2 * embedding_dim  # 2 表示有两种 segment（例如：句子A和句子B）

    # 3. Position Embedding（位置嵌入层）参数
    position_embedding_params = max_position_embeddings * embedding_dim  # 最大位置嵌入维度 * 嵌入维度
    
    # 总 embedding 层参数
    total_embedding_params = token_embedding_params + segment_embedding_params + position_embedding_params

    # 计算 Transformer 层的参数
    # 1. Attention 参数：包括 Q, K, V 矩阵以及它们的偏置
    attention_weights_params = 3 * embedding_dim * embedding_dim + 3 * embedding_dim  # 3个矩阵（Q, K, V）加上每个矩阵的偏置
    
    # 2. 每个 Attention Head 的参数：每个头的 Q, K, V 矩阵
    attention_params_per_head = (embedding_dim // num_attention_heads) * (embedding_dim // num_attention_heads) * 3  # 每个头的 Q, K, V 矩阵参数
    
    # 3. 所有 Transformer 层的 Attention 参数总数
    total_attention_params = (attention_weights_params + attention_params_per_head) * num_transformer_layers  # 每层的 Attention 参数数目乘以层数

    # 计算 Feedforward 层的参数
    # Feedforward 网络的参数量（每层包括两个线性层）
    feedforward_params = (embedding_dim * 4 * embedding_dim + 4 * embedding_dim) * num_transformer_layers  # 第一层的参数 + 偏置
    feedforward_params += (embedding_dim * embedding_dim + embedding_dim) * num_transformer_layers  # 第二层的参数（输出层的参数）

    # 计算总参数数量：embedding 层参数 + transformer 层参数 + feedforward 层参数
    total_params = total_embedding_params + total_attention_params + feedforward_params
    return total_params

# BERT配置参数
vocabulary_size = 30522  # 词汇表大小
embedding_dim = 768  # 隐藏层的维度（也就是嵌入维度）
num_transformer_layers = 12  # Transformer 层的数量
max_position_embeddings = 512  # 最大位置嵌入数

# 计算 BERT 模型的总参数量
total_params = count_bert_params(vocabulary_size, embedding_dim, num_transformer_layers, max_position_embeddings)
print(f"BERT-Base 模型的总参数数量: {total_params}")
