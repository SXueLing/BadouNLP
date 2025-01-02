v_size = 10000  # 假设的词表大小
hidden_size = 768

token_embedding = v_size * hidden_size
segment_embedding = 2 * hidden_size
position_embedding = 512 * hidden_size
embedding = token_embedding + segment_embedding + position_embedding

layer_normalization = 2 * hidden_size

bias = hidden_size
w_q = hidden_size ** 2
w_k = hidden_size ** 2
w_v = hidden_size ** 2
w_o = hidden_size ** 2
self_attention = w_q + w_k + w_v + w_o + bias
feed_forward = (hidden_size * hidden_size * 4 + bias * 4) + ((hidden_size * 4) * hidden_size + bias)
transformer = self_attention + layer_normalization + feed_forward + layer_normalization

bert = (embedding + layer_normalization) + transformer * 12  # 1个embedding层加上12个transformer
print(bert)
