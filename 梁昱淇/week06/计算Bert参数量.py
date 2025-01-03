import numpy as np

# embedding层维度
embedding_dim = 768
# 词表大小(上课时没讲，但根据其他代码来看应该是这个数)
vocab_size = 30522
# segment_embedding仅用于分割句子，维度为2
segment_types = 2
# position_embedding数量(以base版本为例，仅支持最多512个向量)
max_position_length = 512
# 多头机制头的数量
num_heads = 12
# Feed_Forward层第一层输出维度（4为当时的实验值，无特殊含义）
ff_intermediate_dim = 4 * embedding_dim   # （ = 3072）
# 层数
num_layers = 12


# embedding层参数量计算（将token_embedding、position_embedding和segment_embedding相加得到embedding层的参数量）
token_embedding_params = np.multiply(vocab_size, embedding_dim)
position_embedding_params = np.multiply(max_position_length, embedding_dim)
segment_embedding_params = np.multiply(segment_types, embedding_dim)
input_embedding_params = np.sum([token_embedding_params, position_embedding_params, segment_embedding_params])


# 多头注意力机制参数量计算(不用多头的话，QKV都是768*768，所以是768*768*3，分12层就再除以2，之后再加上偏置项)
qkv_params_per_head = np.multiply(np.divide(embedding_dim * embedding_dim, num_heads), 3)
attention_output_params = np.multiply(num_heads, np.multiply(np.divide(embedding_dim, num_heads), embedding_dim))
multi_head_attention_params = np.sum([np.multiply(qkv_params_per_head, num_heads), attention_output_params])


# Add & Norm参数量计算（残差机制，把输入前的X和输入后的Z相加，补充可能损失的信息，其中X和Z都是768）
add_norm_params = np.multiply(embedding_dim, 2)


# 前馈神经网络参数量计算（先扩大维度为768*4过激活函数，然后再还原为768，之后再加上偏置项）
ff_layer1_params = np.add(np.multiply(embedding_dim, ff_intermediate_dim), ff_intermediate_dim)
ff_layer2_params = np.add(np.multiply(ff_intermediate_dim, embedding_dim), embedding_dim)
feed_forward_params = np.sum([ff_layer1_params, ff_layer2_params])


# 总参数量计算
total_params = np.add(input_embedding_params, np.multiply(num_layers, np.sum([multi_head_attention_params, add_norm_params, feed_forward_params, add_norm_params])))


print("输入嵌入层参数量:", input_embedding_params)
print("多头注意力机制参数量:", multi_head_attention_params)
print("Add & Norm参数量:", add_norm_params)
print("前馈神经网络参数量:", feed_forward_params)
print("BERT - Base模型总参数量:", total_params)
