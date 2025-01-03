embedding_dim = 768
x_length = 10
vac = 100

# embedding层
embedding_param = vac * embedding_dim + x_length * embedding_dim
embedding_nor = embedding_param + embedding_dim
# self_attention层
self_attention = (embedding_dim * embedding_dim + embedding_dim)*3
self_attention_norm = embedding_param + embedding_dim
# feed_forward层
ff_param = (embedding_dim * embedding_dim + embedding_dim) *2
ff_norm = embedding_param + embedding_dim

total_param = embedding_param + embedding_nor + self_attention + ff_param + ff_norm

print(total_param)
