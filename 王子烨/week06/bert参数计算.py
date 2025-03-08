# -*- coding: utf-8 -*-
# @Time    : 2025/1/2 18:35
# @Author  : yeye
# @File    : bert层参数.py
# @Software: PyCharm
# @Desc    :
def cal_bert_paramter(vocab_size, hidden_size):
    # embedding层
    embedding_number = cal_embedding(vocab_size, hidden_size)
    # print(f"Embedding层参数量: {embedding_number}")
    # self_attention层
    self_attention = cal_self_attention(hidden_size)
    # print(f"Self-Attention层参数量: {self_attention}")
    # LayerNorm层
    layerNorm_number = cal_LayerNorm(hidden_size)
    # print(f"LayerNorm层参数量: {layerNorm_number}")
    # feed forward层
    feed_forward_number = cal_Feed_Forward(hidden_size)
    # print(f"Feed-Forward层参数量: {feed_forward_number}")
    # LayerNorm层
    layerNorm_number2 = cal_LayerNorm(hidden_size)

    # pool层
    pool_number = cal_pool(hidden_size)
    # print(f"Pooler层参数量: {pool_number}")
    bert_number = embedding_number + self_attention + layerNorm_number + feed_forward_number + layerNorm_number2 + pool_number
    print(bert_number)


def cal_pool(hidden_size):
    w = hidden_size * hidden_size
    b = hidden_size
    return w + b


def cal_Feed_Forward(hidden_size):
    w1 = hidden_size * (hidden_size * 4)
    b1 = hidden_size * 4
    w2 = (hidden_size * 4) * hidden_size
    b2 = hidden_size
    return w1 + w2 + b2 + b1


def cal_LayerNorm(hidden_size):
    ln_weight = hidden_size
    ln_bia = hidden_size
    ln = ln_bia + ln_weight
    return ln


def cal_embedding(vocab_size, hidden_size):
    # embedding层
    # 1. word_embedding
    word_embedding_weight = hidden_size * vocab_size
    # 2. segment_embedding
    segment_embedding_weight = hidden_size * 2
    # 3. position_embedding
    position_embedding_weight = hidden_size * 512
    # 4. LN
    ln_weight = hidden_size
    ln_bia = hidden_size
    embedding_number = word_embedding_weight + segment_embedding_weight + position_embedding_weight + ln_weight + ln_bia
    return embedding_number


def cal_self_attention(hidden_size):
    # 1.query
    query_weight = hidden_size * hidden_size
    query_bias = hidden_size
    # 1.key
    key_weight = hidden_size * hidden_size
    key_bias = hidden_size
    # 1.value
    value_weight = hidden_size * hidden_size
    value_bias = hidden_size
    dense_weight = hidden_size * hidden_size
    dense_bias = hidden_size
    self_attention = query_weight + query_bias + key_bias + key_weight + value_bias + value_weight + dense_bias + dense_weight
    return self_attention


cal_bert_paramter(21128, 768)
from transformers import BertModel

# 载入BERT预训练模型
bert = BertModel.from_pretrained(r"E:\model\bert-base-chinese", return_dict=False)

# 查看BERT的所有参数
state_dict = bert.state_dict()

# 初始化参数计数器
total_params = 0

# 输出每一层的参数数量
for name, param in state_dict.items():
    param_size = param.numel()  # 获取参数的总数
    total_params += param_size  # 累加所有参数

print(total_params)
