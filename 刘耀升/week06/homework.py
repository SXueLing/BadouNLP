def get_bert_params(num_layers, hidden_size = 768, vocab_size = 128):
    # Embedding层参数
    word_embedding = vocab_size * hidden_size
    position_embedding = 512 * hidden_size
    token_type_embedding = 2 * hidden_size
    embedding_layer_norm = hidden_size * 2
    embedding_params = word_embedding + position_embedding + token_type_embedding + embedding_layer_norm

    # Self-Attention层参数 qkv线性层+输出线性层+layer_norm
    q_weight = hidden_size * hidden_size
    q_biase = hidden_size
    k_weight = hidden_size * hidden_size
    k_biase = hidden_size
    v_weight = hidden_size * hidden_size
    v_biase = hidden_size
    out_weight = hidden_size * hidden_size
    out_biase = hidden_size
    attention_layer_norm = 2 * hidden_size
    attention_params = q_weight + q_biase + k_weight + k_biase + v_weight + v_biase + out_weight + out_biase + attention_layer_norm

    # Feed-Forward层参数 线性层1+激活层+线性层2+layer_norm
    feedforward_linear1_weight = hidden_size * 3072
    feedforward_linear1_bias = 3072
    feedforward_linear2_weight = hidden_size * 3072
    feedforward_linear2_bias = hidden_size
    feedforward_layer_norm = hidden_size * 2
    feedforward_params = feedforward_linear1_weight + feedforward_linear1_bias + feedforward_linear2_weight + feedforward_linear2_bias + feedforward_layer_norm

    # Pooler层参数
    pooler_dense_weight = hidden_size * hidden_size
    pooler_dense_bias = hidden_size
    poolar_params = pooler_dense_weight + pooler_dense_bias

    # 总参数量
    total_params = embedding_params + num_layers * (attention_params + feedforward_params) + poolar_params

    return total_params
