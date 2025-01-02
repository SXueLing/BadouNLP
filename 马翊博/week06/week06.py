# 导入所需的库
import torch
from transformers import BertModel


# 计算一个BertLayer的参数量
def compute_bert_layer_params(hidden_size, num_attention_heads, intermediate_size):
    # 多头自注意力机制
    attention_params = (hidden_size * hidden_size * 3) + (hidden_size * hidden_size)

    # 前馈神经网络
    feed_forward_params = (hidden_size * intermediate_size * 2) + (intermediate_size * hidden_size)

    # 总参数量（包括 LayerNorm 和 bias）
    layer_norm_params = 2 * hidden_size * 2
    total_params = attention_params + feed_forward_params + layer_norm_params

    return total_params


# 计算BERT模型的参数量
def compute_bert_model_params(num_layers, hidden_size, num_attention_heads, intermediate_size, vocab_size,
                              max_position_embeddings):
    embedding_params = (vocab_size * hidden_size) + (max_position_embeddings * hidden_size) + (
            hidden_size * hidden_size * 2)

    bert_layer_params = compute_bert_layer_params(hidden_size, num_attention_heads, intermediate_size)
    encoder_params = num_layers * bert_layer_params

    pooler_params = hidden_size * hidden_size + hidden_size

    total_params = embedding_params + encoder_params + pooler_params

    return total_params


# {
#   "architectures": [
#     "BertForMaskedLM"
#   ],
#   "attention_probs_dropout_prob": 0.1,
#   "directionality": "bidi",
#   "hidden_act": "gelu",
#   "hidden_dropout_prob": 0.1,
#   "hidden_size": 768,
#   "initializer_range": 0.02,
#   "intermediate_size": 3072,
#   "layer_norm_eps": 1e-12,
#   "max_position_embeddings": 512,
#   "model_type": "bert",
#   "num_attention_heads": 12,
#   "num_hidden_layers": 1,
#   "pad_token_id": 0,
#   "pooler_fc_size": 768,
#   "pooler_num_attention_heads": 12,
#   "pooler_num_fc_layers": 3,
#   "pooler_size_per_head": 128,
#   "pooler_type": "first_token_transform",
#   "type_vocab_size": 2,
#   "vocab_size": 21128,
#   "num_labels":18
# }
# BERT-base参数设置
num_layers = 12
hidden_size = 768
num_attention_heads = 12
intermediate_size = 3072
vocab_size = 21128
max_position_embeddings = 512

# 计算BERT-base模型的总参数量
total_params = compute_bert_model_params(num_layers, hidden_size, num_attention_heads, intermediate_size, vocab_size,
                                         max_position_embeddings)

# 打印结果
print(f'Total number of parameters in BERT-base model: {total_params:,}')

# 计算BERT模型每一层的参数量
layer_params = compute_bert_layer_params(hidden_size, num_attention_heads, intermediate_size)
print(f'Number of parameters per BERT layer: {layer_params:,}')
