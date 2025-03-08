import math

def calculate_attention_params(hidden_size, num_attention_heads):
    """
    计算自注意力层的参数量。
    """
    attention_head_size = hidden_size // num_attention_heads
    # 自注意力机制参数量计算：Q、K、V的权重矩阵和输出权重矩阵
    return (3 * hidden_size * attention_head_size + hidden_size * hidden_size) * num_attention_heads
def calculate_ffn_params(hidden_size, intermediate_size):
    """
    计算前馈网络的参数量。
    """
    # 前馈网络参数量计算：两层线性变换的权重矩阵
    return 2 * hidden_size * intermediate_size + intermediate_size * hidden_size
def calculate_transformer_layer_params(hidden_size, num_attention_heads, intermediate_size):
    """
    计算单个Transformer层的参数量。
    """
    # 单个Transformer层的参数量 = 自注意力层参数量 + 前馈网络参数量 + 层归一化参数量
    return calculate_attention_params(hidden_size, num_attention_heads) + calculate_ffn_params(hidden_size, intermediate_size) + 4 * hidden_size
def calculate_embedding_params(vocab_size, hidden_size, max_len):
    """
    计算嵌入层的参数量。
    """
    # 嵌入层参数量计算：词嵌入矩阵 + 位置编码矩阵 + 层归一化参数量
    return vocab_size * hidden_size + max_len * hidden_size + 2 * hidden_size
def calculate_total_params(vocab_size, hidden_size, num_layers, num_attention_heads, max_len, intermediate_size):
    """
    计算Transformer模型的总参数量。
    """
    # Transformer模型的总参数量 = Transformer层的参数量 * 层数 + 嵌入层的参数量
    return num_layers * calculate_transformer_layer_params(hidden_size, num_attention_heads, intermediate_size) + calculate_embedding_params(vocab_size, hidden_size, max_len)
# 设置模型参数
vocab_size = 21128
hidden_size = 768
num_layers = 12
num_attention_heads = 12
max_len = 512
intermediate_size = 3072  # 通常为隐藏层大小的4倍
# 计算并打印总参数量
total_params = calculate_total_params(vocab_size, hidden_size, num_layers, num_attention_heads, max_len, intermediate_size)
print(f"总参数量：{total_params / (1024 * 1024):.2f}M")
