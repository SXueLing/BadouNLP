hidden_size = 768   # 维度大小
word_size = 4 # 词表大小
num_layers = 1 # 层数
def parameter_size(hidden_size, word_size, num_layers):
    embdding_size = 3*word_size*hidden_size  # embedding 层训练参数，3层数据
    self_attention_size = num_layers*(hidden_size*hidden_size*3+hidden_size*3)  # 自注意力 3个线性层
    feed_forward_size = word_size*hidden_size*hidden_size*2 + word_size*hidden_size + hidden_size  # 两个线性层
    par_size_num = embdding_size + self_attention_size + feed_forward_size
    return par_size_num
per_size = parameter_size(hidden_size, word_size, num_layers)

print(per_size)
