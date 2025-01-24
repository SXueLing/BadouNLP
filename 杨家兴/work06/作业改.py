# [{'word_embeddings': (21128, 768)}, {'position_embeddings': (512, 768)}, 
#  {'token_type_embeddings': (2, 768)}, {'embeddings_layer_norm_weight': (768,)},
#    {'embeddings_layer_norm_bias': (768,)}, {'q_w1': (768, 768)}, {'q_b1': (768,)},
#      {'k_w1': (768, 768)}, {'k_b1': (768,)}, {'v_w1': (768, 768)}, {'v_b1': (768,)},
#        {'attention_output_weight1': (768, 768)}, {'attention_output_bias1': (768,)}, 
#        {'attention_layer_norm_w1': (768,)}, {'attention_layer_norm_b1': (768,)}, 
#        {'intermediate_weight1': (3072, 768)}, {'intermediate_bias1': (3072,)}, 
#        {'output_weight1': (768, 3072)}, {'output_bias1': (768,)}, {'ff_layer_norm_w1': (768,)},
#          {'ff_layer_norm_b1': (768,)}, {'pooler_dense_weight': (768, 768)},
#            {'pooler_dense_bias': (768,)}] 权重shape
import torch
import math
import torch.nn as nn
import numpy as np
from transformers import BertModel

model = BertModel.from_pretrained(r"/Users/mac/Documents/bert-base-chinese", return_dict=False)

total = [p.numel() for p in model.parameters()]
print(total, 'total')
print('model参数总和：', sum(total))

vocab_length = 21128
hidden_size = 768
token_size = 2 # token embedding
position_size = 512 # position embedding
num_layers = 2 # config配置的transform层数
feed_size = 3072 # 前馈网络

# 词向量做embedding。position embedding表示语序，输入的是[0,1,2,3,4]
embeddings = vocab_length * hidden_size + token_size * hidden_size + position_size * hidden_size + hidden_size + hidden_size

# attention_output_weight是qkv处理完以后做一次线性变换
self_attention = hidden_size * hidden_size * 3 + hidden_size * 3 + hidden_size * hidden_size + hidden_size + hidden_size + hidden_size

# 前馈网络
feed_forward = hidden_size * feed_size + feed_size + feed_size * hidden_size + hidden_size

# 出前馈网络以后做一次残差和归一化
layer_norm = hidden_size + hidden_size

pooler = hidden_size * hidden_size + hidden_size # 出来以后还是 v * hidden_size

my_result = embeddings + num_layers * (self_attention + feed_forward + layer_norm) + pooler
print('自己计算的结果：', my_result)
# model参数总和： 31388928
# 自己计算的结果： 31388928



