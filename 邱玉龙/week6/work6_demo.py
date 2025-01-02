#  -*- coding: utf-8 -*-
"""
Author: loong
Time: 2025/1/2 21:49
File: work6_demo.py
Software: PyCharm
"""
import numpy as np
import math

import torch
from transformers import BertModel
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)

#gelu激活函数
def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))

#对比模型
bert_obj = BertModel.from_pretrained(r"./bert-base-chinese", return_dict=False)
# odict_keys(['embeddings.word_embeddings.weight', 'embeddings.position_embeddings.weight', 'embeddings.token_type_embeddings.weight', 'embeddings.LayerNorm.weight', 'embeddings.LayerNorm.bias', 'encoder.layer.0.attention.self.query.weight', 'encoder.layer.0.attention.self.query.bias', 'encoder.layer.0.attention.self.key.weight', 'encoder.layer.0.attention.self.key.bias', 'encoder.layer.0.attention.self.value.weight', 'encoder.layer.0.attention.self.value.bias', 'encoder.layer.0.attention.output.dense.weight', 'encoder.layer.0.attention.output.dense.bias', 'encoder.layer.0.attention.output.LayerNorm.weight', 'encoder.layer.0.attention.output.LayerNorm.bias', 'encoder.layer.0.intermediate.dense.weight', 'encoder.layer.0.intermediate.dense.bias', 'encoder.layer.0.output.dense.weight', 'encoder.layer.0.output.dense.bias', 'encoder.layer.0.output.LayerNorm.weight', 'encoder.layer.0.output.LayerNorm.bias', 'pooler.dense.weight', 'pooler.dense.bias'])
state_dict = bert_obj.state_dict()
bert_obj.eval()

a = [1111,2222,3333,4444]
x = np.array(a)  # 向量
torch_x = torch.LongTensor([x])
class Bert_b():
    def __init__(self, state_dict):
        self.mutil_attention_heads = 12
        self.hidden_size = 768
        self.cnt_transformer = 1
        self.load_weights(state_dict)

    def load_weights(self, state_dict):
        # 加载权重
        # odict_keys(['embeddings.word_embeddings.weight', 'embeddings.position_embeddings.weight',
        # 'embeddings.token_type_embeddings.weight',
        # 'embeddings.LayerNorm.weight', 'embeddings.LayerNorm.bias',
        # 'encoder.layer.0.attention.self.query.weight',
        # 'encoder.layer.0.attention.self.query.bias',
        # 'encoder.layer.0.attention.self.key.weight',
        # 'encoder.layer.0.attention.self.key.bias',
        # 'encoder.layer.0.attention.self.value.weight',
        # 'encoder.layer.0.attention.self.value.bias',
        # 'encoder.layer.0.attention.output.dense.weight',
        # 'encoder.layer.0.attention.output.dense.bias',
        # 'encoder.layer.0.attention.output.LayerNorm.weight',
        # 'encoder.layer.0.attention.output.LayerNorm.bias',
        # 'encoder.layer.0.intermediate.dense.weight',
        # 'encoder.layer.0.intermediate.dense.bias',
        # 'encoder.layer.0.output.dense.weight',
        # 'encoder.layer.0.output.dense.bias',
        # 'encoder.layer.0.output.LayerNorm.weight',
        # 'encoder.layer.0.output.LayerNorm.bias',
        # 'pooler.dense.weight',
        # 'pooler.dense.bias'])

        self.word_embedding_w = state_dict['embeddings.word_embeddings.weight'].numpy()
        self.position_embedding_w = state_dict['embeddings.position_embeddings.weight'].numpy()
        self.token_type_embedding_w = state_dict['embeddings.token_type_embeddings.weight'].numpy()
        self.layernorm_w = state_dict['embeddings.LayerNorm.weight'].numpy()
        self.layernorm_b = state_dict['embeddings.LayerNorm.bias'].numpy()
        print("word_embedding_w ",self.word_embedding_w.shape)
        print("position_embedding_w ",self.position_embedding_w.shape)
        print("token_type_embedding_w ",self.token_type_embedding_w.shape)
        print("layernorm_w ",self.layernorm_w.shape)
        print("layernorm_b ",self.layernorm_b.shape)

        self.pooler_dense_w = state_dict['pooler.dense.weight'].numpy()
        self.pooler_dense_b = state_dict['pooler.dense.bias'].numpy()
        print("pooler_dense_w ", self.pooler_dense_w.shape)
        print("pooler_dense_b ", self.pooler_dense_b.shape)
        # 多层transformer
        self.transformer_weights = list()
        for i in range(self.cnt_transformer):
            q_w = state_dict[f'encoder.layer.{i}.attention.self.query.weight'].numpy()
            k_w = state_dict[f'encoder.layer.{i}.attention.self.key.weight'].numpy()
            v_w = state_dict[f'encoder.layer.{i}.attention.self.value.weight'].numpy()
            q_b = state_dict[f'encoder.layer.{i}.attention.self.query.bias'].numpy()
            k_b = state_dict[f'encoder.layer.{i}.attention.self.key.bias'].numpy()
            v_b = state_dict[f'encoder.layer.{i}.attention.self.value.bias'].numpy()
            print("q_w ",q_w.shape)
            print("k_w ",k_w.shape)
            print("v_w ",v_w.shape)
            print("q_b ",q_b.shape)
            print("k_b ",k_b.shape)
            print("v_b ",v_b.shape)
            dense_weight = state_dict[f'encoder.layer.{i}.attention.output.dense.weight'].numpy()
            print("dense_weight ",dense_weight.shape)

            dense_bias = state_dict[f'encoder.layer.{i}.attention.output.dense.bias'].numpy()
            print("dense_bias ",dense_bias.shape)

            LayerNorm_weight = state_dict[f'encoder.layer.{i}.attention.output.LayerNorm.weight'].numpy()
            print("LayerNorm_weight ",LayerNorm_weight.shape)

            LayerNorm_bias = state_dict[f'encoder.layer.{i}.attention.output.LayerNorm.bias'].numpy()
            print("LayerNorm_bias ",LayerNorm_bias.shape)

            intermediate_dense_weight = state_dict[f'encoder.layer.{i}.intermediate.dense.weight'].numpy()
            print("intermediate_dense_weight ",intermediate_dense_weight.shape)

            intermediate_dense_bias = state_dict[f'encoder.layer.{i}.intermediate.dense.bias'].numpy()
            print("intermediate_dense_bias ",intermediate_dense_bias.shape)

            output_dense_weight = state_dict[f'encoder.layer.{i}.output.dense.weight'].numpy()
            print("output_dense_weight ",output_dense_weight.shape)

            output_dense_bias = state_dict[f'encoder.layer.{i}.output.dense.bias'].numpy()
            print("output_dense_bias ",output_dense_bias.shape)

            output_LayerNorm_weight = state_dict[f'encoder.layer.{i}.output.LayerNorm.weight'].numpy()
            print("output_LayerNorm_weight ",output_LayerNorm_weight.shape)

            output_LayerNorm_bias = state_dict[f'encoder.layer.{i}.output.LayerNorm.bias'].numpy()
            print("output_LayerNorm_bias ",output_LayerNorm_bias.shape)
            self.transformer_weights.append([q_w, q_b, k_w, k_b, v_w, v_b, dense_weight, dense_bias, LayerNorm_weight, LayerNorm_bias, intermediate_dense_weight, intermediate_dense_bias, output_dense_weight, output_dense_bias, output_LayerNorm_weight, output_LayerNorm_bias])



    def embeddings_func(self,x):
        # embedding层
        w_e = self.get_embedding(self.word_embedding_w, x)

        # position embedding
        p_e = self.get_embedding(self.position_embedding_w, np.array(list(range(len(x)))))
        # token embedding
        t_e = self.get_embedding(self.token_type_embedding_w, np.array([0] * len(x)))
        embeddings = w_e + p_e + t_e
        return self.layer_norm(embeddings, self.layernorm_w, self.layernorm_b)

    def attention_func(self,x ,q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias, mutil_head, hide_size):
        # 注意力层
        q = np.dot(x, q_w.T) + q_b
        k = np.dot(x, k_w.T) + k_b
        v = np.dot(x, v_w.T) + v_b
        # 每头多少
        attention_one_size = int(hide_size / mutil_head)

        q = self.transpose_for_scores(q, attention_one_size, mutil_head)
        k = self.transpose_for_scores(k, attention_one_size, mutil_head)
        v = self.transpose_for_scores(v, attention_one_size, mutil_head)

        qk = np.matmul(q, k.swapaxes(1, 2))
        qk /= np.sqrt(attention_one_size)
        attention_output = softmax(qk)
        qkv = np.matmul(attention_output, v)
        qkv = qkv.swapaxes(0, 1).reshape(-1, hide_size)
        attention = np.dot(qkv, attention_output_weight.T) + attention_output_bias
        return attention

    def feed_forward(self,x, intermediate_dense_weight, intermediate_bias, output_weight, output_bias):
        # 前向传播层
        x = np.dot(x, intermediate_dense_weight.T) + intermediate_bias
        x = gelu(x)
        x = np.dot(x, output_weight.T) + output_bias
        return x



    def transpose_for_scores(self, x, attention_head_size, num_attention_heads):
        max_len, hidden_size = x.shape
        x = x.reshape(max_len, num_attention_heads, attention_head_size)
        x = x.swapaxes(1, 0)
        return x
    def get_embedding(self, embedding_matrix, x):
        return np.array([embedding_matrix[index] for index in x])

    def layer_norm(self, x, w, b):
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        x = x * w + b
        return x

    def one_transformer_layer_forward(self, x, index):
        one_w = self.transformer_weights[index]
        q_w, q_b, k_w, k_b, v_w, v_b, dense_weight, dense_bias, LayerNorm_weight, LayerNorm_bias, intermediate_dense_weight, intermediate_dense_bias, output_dense_weight, output_dense_bias, output_LayerNorm_weight, output_LayerNorm_bias = one_w

        attention_output = self.attention_func(x,
                                q_w, q_b,
                                k_w, k_b,
                                v_w, v_b,
                                dense_weight, dense_bias,
                                self.mutil_attention_heads,
                                self.hidden_size)
        x = self.layer_norm(x + attention_output, LayerNorm_weight, LayerNorm_bias)
        feed_forward_output = self.feed_forward(x,intermediate_dense_weight,intermediate_dense_bias,output_dense_weight,output_dense_bias)
        x = self.layer_norm(x + feed_forward_output, output_LayerNorm_weight, output_LayerNorm_bias)
        return x

    def pooler_output_layer(self, x):
        x = np.dot(x, self.pooler_dense_w.T) + self.pooler_dense_b
        x = np.tanh(x)
        return x

    def all_transformer_layer_forward(self, embedding_x):
        for i in range(self.cnt_transformer):
            embedding_x = self.one_transformer_layer_forward(embedding_x, i)
        return embedding_x
    def start_transformer(self, x):
        embedding_x = self.embeddings_func(x)
        sequence_output =self.all_transformer_layer_forward(embedding_x)
        pooler_output = self.pooler_output_layer(sequence_output[0])
        return sequence_output, pooler_output



db = Bert_b(state_dict)
diy_sequence_output, diy_pooler_output = db.start_transformer(x)
#torch
torch_sequence_output, torch_pooler_output = bert_obj(torch_x)

# print(diy_sequence_output)
# print(torch_sequence_output)

"""

embedding
词表 +  POSITION  +  TOKEN_TYPE +  LayerNorm_W + LayerNorm_B
词参数 : (词表长度 + 512 + 2 + 2) * 768 

TRANSFORMER
    一个attention +   归一化     +  前馈传播 (2个线性层 + 一个激活函数)                            + 一个归一化
N * (3 * 768 ** 2) + (5 * 768) +  ((3072 * 768) + 3072) + (（3072 *768） + 768) + (768 * 2)  + ((768 *2) + 768)

char_dim 维度
N 多少tansformer 层
公式：
(词表长度+516) * char_dim + N *(3 * char_dim ** 2 + 6155 * char_dim+3072)

"""
