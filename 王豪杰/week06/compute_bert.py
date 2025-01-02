def calculate_bert_params(vocab_size, max_position_embeddings, hidden_size, num_attention_heads, num_layers):
    # 1. 词嵌入参数 (包括词汇表和位置嵌入)
    embedding_params = vocab_size * hidden_size + max_position_embeddings * hidden_size
    
    # 2. 每层Transformer参数 (Self-Attention + Feed Forward)
    # Self-Attention部分
    attention_params_per_layer = 3 * hidden_size * hidden_size * num_attention_heads
    
    # Feed Forward部分 (两个全连接层)
    feedforward_params_per_layer = 2 * hidden_size * 4 * hidden_size
    
    # 每层的总参数量
    layer_params = attention_params_per_layer + feedforward_params_per_layer
    
    # 总的Transformer层参数量
    transformer_params = layer_params * num_layers
    
    # 3. 输出层参数 (分类任务)
    output_params = hidden_size * 2  # 假设是二分类任务，可以根据任务调整输出数量
    
    # 总参数量
    total_params = embedding_params + transformer_params + output_params
    
    return total_params

# BERT-Base参数（典型配置）
vocab_size = 30522  # 词汇表大小
max_position_embeddings = 512  # 最大序列长度
hidden_size = 768  # 隐藏层维度
num_attention_heads = 12  # 注意力头数
num_layers = 12  # Transformer层数

params = calculate_bert_params(vocab_size, max_position_embeddings, hidden_size, num_attention_heads, num_layers)
print(f'BERT-Base模型的参数量: {params / 1e6:.2f}M')
