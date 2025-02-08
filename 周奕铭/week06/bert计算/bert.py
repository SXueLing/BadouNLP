## 计算 BERT中 有多少参数

def calculate_bert_12layer_parameters():
    # 词汇表大小，假设为 30522
    vocab_size = 30522
    # 隐藏层大小，BERT - Base 为 768
    hidden_size = 768
    # 多头注意力的头数，BERT - Base 为 12
    num_heads = 12
    # 层数，这里为 12
    num_layers = 12
    # 全连接层的维度，通常是 hidden_size * 4
    intermediate_size = hidden_size * 4
    # 最大位置编码数，假设为 512
    max_position_embeddings = 512
    # 段嵌入层大小，因为有两个段，通常用于区分句子 A 和句子 B
    num_segments = 2

    # 词嵌入层参数，考虑词汇表大小和隐藏层大小
    embedding_params = vocab_size * hidden_size
    # 位置嵌入层参数，考虑最大位置编码数和隐藏层大小
    position_embedding_params = max_position_embeddings * hidden_size
    # 段嵌入层参数，考虑段的数量和隐藏层大小
    segment_embedding_params = num_segments * hidden_size

    # 每个 Transformer 层的参数计算
    # 注意力头大小
    attention_head_size = hidden_size // num_heads
    # Q、K、V 矩阵的参数，每个矩阵是 hidden_size * attention_head_size
    attention_qkv_params = 3 * hidden_size * attention_head_size
    # 多头注意力输出层参数，考虑隐藏层大小
    attention_output_params = hidden_size * hidden_size
    # 全连接层的参数，考虑隐藏层大小和中间层大小
    fc_params = hidden_size * intermediate_size + intermediate_size * hidden_size
    # 每层的参数，包含多头注意力和全连接层，考虑偏置项
    layer_params = (attention_qkv_params + attention_output_params + fc_params) * num_heads + hidden_size * 3
    # 所有层的参数，考虑层数
    all_layers_params = num_layers * layer_params

    # 总参数，包含嵌入层和所有层的参数
    total_params = embedding_params + position_embedding_params + segment_embedding_params + all_layers_params
    return total_params


if __name__ == "__main__":
    total_params = calculate_bert_12layer_parameters()
    print(f"12 层 BERT 的总参数数量为：{total_params}")
