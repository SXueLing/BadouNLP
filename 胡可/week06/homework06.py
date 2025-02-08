def calculate_bert_parameters(vocab_size, embedding_size, max_position_embeddings, n, hidden_size, num_layers):
    # Embedding层参数量
    embedding_params = vocab_size * embedding_size + max_position_embeddings * embedding_size + n * embedding_size + 2 * embedding_size

    # Transformer层参数量
    transformer_layer_params = 3 * embedding_size * embedding_size + embedding_size * embedding_size + 2 * embedding_size + embedding_size * hidden_size + hidden_size * embedding_size + 2 * embedding_size

    # 总Transformer层参数量
    total_transformer_params = num_layers * transformer_layer_params

    # Pooler层参数量
    pooler_params = embedding_size * embedding_size + embedding_size

    # 总参数量
    total_params = embedding_params + total_transformer_params + pooler_params
    return total_params

# 假设BERT模型的参数
vocab_size = 21128
embedding_size = 768
max_position_embeddings = 512
n = 2
hidden_size = 3072
num_layers = 12

# 计算参数量
total_params = calculate_bert_parameters(vocab_size, embedding_size, max_position_embeddings, n, hidden_size, num_layers)
print(f'Total parameters in BERT: {total_params}')
