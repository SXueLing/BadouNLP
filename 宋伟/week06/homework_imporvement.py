# -*- coding: utf-8 -*-
# @Date    :2025-02-08 13:25:19
# @Author  :SongWei (songweiof@gmail.com)
# @Link    :https://github.com/Finder12345
# @Version :python 3.11
# @Software:Sublime Text


import torch
import math
import torch.nn as nn
from transformers import BertModel

model = BertModel.from_pretrained(r"J:\\八斗课堂学习\\第六周 语言模型\\bert-base-chinese\\bert-base-chinese",return_dict=False)
n = 2 # 句子的个数,当前句子或非当前句子
vocab = 21128  # 词表数目
max_sequence_length = 512 # 句子的最大长度
embedding_size = 768 # 映射空间的维度
hide_size = 3072 # 隐藏层的维度
num_layers = 1  # 隐藏层个数


# embedding过程中的参数：token，positions,segment，embedding
embedding_parametres = vocab*embedding_size+embedding_size+ \
                        max_sequence_length*embedding_size+embedding_size+ \
                        n*embedding_size
                        

# self-attention过程中的参数
self_attention_parameters = (embedding_size*embedding_size+embedding_size)*3


# self-attention-out过程中的参数
self_attention_out_parameters = embedding_size*embedding_size+embedding_size+\
                                embedding_size+embedding_size

# feed-forward 过程参数
feed_forward_parameters = embedding_size*hide_size+hide_size+\
                        hide_size*embedding_size+embedding_size+\
                        embedding_size+ embedding_size

# pool-fc层参数
pooler_fc_parameters = embedding_size*embedding_size+embedding_size
# 模型的总参数
all_parameters = embedding_parametres+\
                (self_attention_parameters+self_attention_out_parameters+feed_forward_parameters)*num_layers+\
                pooler_fc_parameters

print("模型编码区实际参数个数%d" % sum(p.numel() for p in model.parameters()))
print("diy计算的模型参数个数为：%d" % all_parameters)