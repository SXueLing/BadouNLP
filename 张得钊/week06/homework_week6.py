
import numpy as np
import torch
from transformers import BertModel
import math


'''

作业: 计算bert参数量

bert参数量: 
(vocab_size + 512 + 2 + 1 + 1) * hidden_size
+ 12 * ( (4*hidden_size + 2*intermediate_size + 9)*hidden_size + intermediate_size )
+ (hidden_size * hidden_size + hidden_size)

vocab_size: 词表大小, bert为21128
hidden_size: 词向量长度, bert中为768
intermediate_size: feed forward层中间层大小, bert中为3072

上述公式计算结果为：102267648

具体过程看下面的详细解释，后面的代码实现验证了正确性

'''

# 此处是详细解释：
'''
bert: embedding层 + transformer层(12层) + pooler层

embedding层: word_embeddings + position_embeddings + segment_embeddings + layer_normalization
(三层+归一化)
word_embeddings: vocab_size * hidden_size
position_embeddings: 512 * hidden_size
segment_embeddings: 2 * hidden_size
layer_norm_W: 1 * hidden_size,  layer_norm_b: 1 * hidden_size
参数量:
vocab_size * hidden_size + 512 * hidden_size + 2 * hidden_size + 1 * hidden_size + 1 * hidden_size
= (vocab_size + 512 + 2 + 1 + 1) * hidden_size

单个transformer层: Attention(Q,K,V) + Linear + layer_norm + linear + gelu + linear + layer_norm
Linear(Attention(Q,K,V)) -> layer_normalization(x_embedding+x_attention)
-> Linear(gelu(Linear(x))) -> layer_normalization(x_embedding+x_attention)
Q_W: hidden_size * hidden_size,  Q_b: 1 * hidden_size
K_W: hidden_size * hidden_size,  K_b: 1 * hidden_size
V_W: hidden_size * hidden_size,  V_b: 1 * hidden_size
Linear_W: hidden_size * hidden_size,  Linear_b: 1 * hidden_size
layer_norm1_W: 1 * hidden_size,  layer_norm1_b: 1 * hidden_size
ff_Linear1_W: intermediate_size * hidden_size,  ff_Linear1_b: 1 * intermediate_size
ff_Linear2_W: hidden_size * intermediate_size,  ff_Linear2_b: 1 * hidden_size
layer_norm2_W: 1 * hidden_size,  layer_norm2_b: 1 * hidden_size
参数量:
4 * (hidden_size * hidden_size + 1 * hidden_size)
+ 1 * hidden_size + 1 * hidden_size
+ intermediate_size * hidden_size + intermediate_size + hidden_size * intermediate_size + 1 * hidden_size
+ 1 * hidden_size + 1 * hidden_size
= (4*hidden_size + 4 + 1 + 1 + 2*intermediate_size + 1 + 1 + 1)*hidden_size + intermediate_size
= (4*hidden_size + 2*intermediate_size + 9)*hidden_size + intermediate_size

pooler层:
pooler_W: hidden_size * hidden_size,  pooler_b: 1 * hidden_size

总参数量:
(vocab_size + 512 + 2 + 1 + 1) * hidden_size
+ 12 * ( (4*hidden_size + 2*intermediate_size + 9)*hidden_size + intermediate_size )
+ (hidden_size * hidden_size + hidden_size)

'''

# 此处为代码实现
def compute_bert_param_num():
    vocab_size = 21128 #词表大小
    hidden_size = 768  #词向量维度
    intermediate_size = 3072  #feed forward层的中间层大小
    # 嵌入层部分的参数量
    layer_embedding_size = (vocab_size + 512 + 2 + 1 + 1) * hidden_size
    # transformer部分参数量
    layer_transformer_size = 12 * ( (4*hidden_size + 2*intermediate_size + 9)*hidden_size + intermediate_size )
    # pooler部分参数量
    layer_pooler_size = hidden_size * hidden_size + hidden_size
    result = layer_embedding_size + layer_transformer_size + layer_pooler_size
    return result

def get_bert_param_num():
    bert = BertModel.from_pretrained("bert-base-chinese", return_dict=False)
    state_dict = bert.state_dict()
    num_param = 0
    for key in state_dict:
        param = state_dict[key]
        # 计算参数数量
        num_param += param.numel()
    return num_param

num_param_true = get_bert_param_num()
num_param_compute = compute_bert_param_num()
print('bert实际参数量: ', num_param_true)
print('计算得到的bert参数量: ', num_param_compute)
