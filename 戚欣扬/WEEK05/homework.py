import math
def cal_params(vocab_size, hidden_size, num_layers, max_len):
    transformer_layer_params = 12 * hidden_size**2 + 16 * hidden_size
    embedding_params = vocab_size * hidden_size + max_len * hidden_size + 2 * hidden_size
    total_params = num_layers * transformer_layer_params + embedding_params
    return total_params

vocab_size = 21128
hidden_size = 768
max_len = 512
num_layers = 12

total_params = cal_params(vocab_size, hidden_size, num_layers, max_len)
print(f"总参数量：{total_params // (1024*1024)}M")
