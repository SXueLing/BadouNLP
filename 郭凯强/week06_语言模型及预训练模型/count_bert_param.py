def calculate_bert_params(
    vocab_size=30522,
    hidden_size=768,
    num_layers=12,
    num_attention_heads=12,
    max_position_embeddings=512,
    intermediate_size=3072
):
    # Embedding层参数
    embedding_params = (
        vocab_size * hidden_size +  # Token Embeddings
        max_position_embeddings * hidden_size +  # Position Embeddings
        2 * hidden_size +  # Segment Embeddings
        2 * hidden_size  # Layer Norm
    )
    
    # 每个Transformer层参数
    transformer_layer_params = (
        # Self-Attention
        3 * hidden_size * hidden_size +  # Q,K,V
        hidden_size * hidden_size +  # 输出投影
        2 * hidden_size +  # Layer Norm
        
        # Feed Forward
        hidden_size * intermediate_size +  # 第一个线性层
        intermediate_size * hidden_size +  # 第二个线性层
        2 * hidden_size  # Layer Norm
    )
    
    # 总参数量
    total_params = embedding_params + num_layers * transformer_layer_params
    
    return {
        'embedding_params': embedding_params,
        'per_transformer_params': transformer_layer_params,
        'total_params': total_params
    }

# 计算BERT-base参数量
params = calculate_bert_params()
print(f"Embedding层参数量: {params['embedding_params']:,}")
print(f"每个Transformer层参数量: {params['per_transformer_params']:,}")
print(f"总参数量: {params['total_params']:,}")