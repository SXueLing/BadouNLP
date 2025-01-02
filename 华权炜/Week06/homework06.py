import torch
import math
import torch.nn as nn
import numpy as np
from transformers import BertModel

model = BertModel.from_pretrained(path, return_dict=False)
#vocab_size = 按照具体的模型情况而定
#type_vocab_size = 按照具体的模型情况而定
#max_position_embeddings = 按照具体的模型情况而定
#hidden_size = 按照具体的模型情况而定
#intermediate_size = 按照具体的模型情况而定


def bert_parameters():
  token_embeddings = vocab_size * hidden_size
  segment_embedding = type_vocab_size * hidden_size
  position_embeddings = max_position_embeddings * hidden_size
  layer_norm1 = hidden_size + hidden_size
  self_attention_parameters = (hidden_size * hidden_size + hidden_size) * 3
  self_attention_out_parameters = hidden_size * hidden_size + hidden_size
  layer_norm2 = hidden_size + hidden_size
  feed_forward = (hidden_size * intermediate_size + intermediate_size) * 2
  layer_norm3 = hidden_size + hidden_size
  pool_fc_parameters = hidden_size * hidden_size + hidden_size
  return token_embeddings + segment_embedding + position_embeddings + layer_norm1 + \
         (self_attention_parameters + self_attention_out_parameters + layer_norm2 + feed_forward + layer_norm3) * num_layers \ 
         + pool_fc_parameters

print('模型的参数个数为：%d'%bert_parameters())
