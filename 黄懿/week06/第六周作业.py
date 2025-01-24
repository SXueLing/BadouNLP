import json

# 读取JSON文件
with open('config.json', 'r') as f:
    params = json.load(f)


# 使用导入的参数
hidden_size = params["hidden_size"]
intermediate_size = params["intermediate_size"]
max_position_embeddings = params["max_position_embeddings"]
num_attention_heads = params["num_attention_heads"]
num_hidden_layers = params["num_hidden_layers"]
type_vocab_size = params["type_vocab_size"]
vocab_size = params["vocab_size"]


token_embeddings = vocab_size * hidden_size
position_embeddings = max_position_embeddings * hidden_size
segment_embeddings = type_vocab_size * hidden_size
embeddings_weight_layer_norm = hidden_size * 2
num_embeddings = token_embeddings + position_embeddings + segment_embeddings + embeddings_weight_layer_norm

self_attention = hidden_size * hidden_size/num_attention_heads * 3 * num_attention_heads + hidden_size * hidden_size
self_attention_weight_layer_norm = hidden_size * 2
num_self_attention = (self_attention + self_attention_weight_layer_norm) * num_hidden_layers

feed_forward_1 = hidden_size * intermediate_size + intermediate_size
feed_forward_2 = intermediate_size * hidden_size + hidden_size
feed_forward = feed_forward_1 + feed_forward_2
feed_forward_weight_layer_norm = hidden_size * 2
num_feed_forward = (feed_forward + feed_forward_weight_layer_norm) * num_hidden_layers

num_params = num_embeddings + num_self_attention + num_feed_forward

print("BERT参数量：", int(num_params))
