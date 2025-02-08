import json
import torch
from transformers import BertModel


def compute_bert_params(config_file):
    # 从配置文件中读取参数
    with open(config_file, 'r') as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    num_attention_heads = config["num_attention_heads"]
    intermediate_size = config["intermediate_size"]
    vocab_size = config["vocab_size"]
    max_position_embeddings = config["max_position_embeddings"]
    num_layers = config["num_hidden_layers"]

    # 计算词嵌入层参数量
    embedding_params = (vocab_size * hidden_size) + (max_position_embeddings * hidden_size) + (
            hidden_size * hidden_size * 2)

    # 计算多头自注意力机制参数量
    attention_params = (hidden_size * hidden_size * 3) + (hidden_size * hidden_size)

    # 计算前馈神经网络参数量
    feed_forward_params = (hidden_size * intermediate_size * 2) + (intermediate_size * hidden_size)

    # 计算 LayerNorm 和 bias 参数量
    layer_norm_params = 2 * hidden_size * 2

    # 计算一层的总参数量
    layer_params = attention_params + feed_forward_params + layer_norm_params

    # 计算整个编码器（多层）的参数量
    encoder_params = num_layers * layer_params

    # 计算池化层参数量
    pooler_params = hidden_size * hidden_size + hidden_size

    # 计算模型总参数量
    total_params = embedding_params + encoder_params + pooler_params

    return total_params, layer_params


# BERT 模型配置文件路径（根据实际情况修改）
config_file = "bert_config.json"

# 计算 BERT 模型的参数量
total_params, layer_params = compute_bert_params(config_file)

# 打印结果
print(f'Total number of parameters in BERT model: {total_params:,}')
print(f'Number of parameters per BERT layer: {layer_params:,}')
