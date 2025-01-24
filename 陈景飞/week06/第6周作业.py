import torch
import math
import numpy as np
from transformers import BertModel

'''

计算BERT-Base里面有多少个可训练的参

'''


# 方法一：直接看bert.state_dict()
def calcu_all_parameter_cnt1():
    bert = BertModel.from_pretrained(r"F:\Desktop\work_space\pretrain_models\bert-base-chinese", return_dict=False)
    state_dict = bert.state_dict()
    # print(state_dict['embeddings.word_embeddings.weight'].shape) #torch.Size([21128, 768])
    # print(state_dict['embeddings.position_embeddings.weight'].shape) #torch.Size([512, 768])
    # print(state_dict['embeddings.token_type_embeddings.weight'].shape) #torch.Size([2, 768])
    # print(state_dict['pooler.dense.weight'].shape)  # torch.Size([768, 768]) #最后一个线性层
    # print(state_dict['pooler.dense.bias'].shape)  # torch.Size([768]) #最后一个线性层

    # print(bert.state_dict().keys())  #查看所有的权值矩阵名称
    # 'embeddings.word_embeddings.weight',
    # 'embeddings.position_embeddings.weight',
    # 'embeddings.token_type_embeddings.weight',
    # 'embeddings.LayerNorm.weight', 'embeddings.LayerNorm.bias',
    # 'encoder.layer.0.attention.self.query.weight', 'encoder.layer.0.attention.self.query.bias',
    # 'encoder.layer.0.attention.self.key.weight', 'encoder.layer.0.attention.self.key.bias',
    # 'encoder.layer.0.attention.self.value.weight', 'encoder.layer.0.attention.self.value.bias',
    # 'encoder.layer.0.attention.output.dense.weight', 'encoder.layer.0.attention.output.dense.bias',
    # 'encoder.layer.0.attention.output.LayerNorm.weight', 'encoder.layer.0.attention.output.LayerNorm.bias',
    # 'encoder.layer.0.intermediate.dense.weight', 'encoder.layer.0.intermediate.dense.bias',
    # 'encoder.layer.0.output.dense.weight', 'encoder.layer.0.output.dense.bias',
    # 'encoder.layer.0.output.LayerNorm.weight', 'encoder.layer.0.output.LayerNorm.bias',
    # 'pooler.dense.weight', 'pooler.dense.bias'

    # 计算embeddings的参数量
    embeddings_word_embeddings_weight_para_cnt = state_dict['embeddings.word_embeddings.weight'].numel()  # (V, 768) V= 21128,
    embeddings_position_embeddings_weight_para_cnt = state_dict['embeddings.position_embeddings.weight'].numel()  # (512, 768)
    embeddings_token_type_embeddings_weight_para_cnt = state_dict['embeddings.token_type_embeddings.weight'].numel()  # (2, 768)
    embeddings_LayerNorm_weight_para_cnt = state_dict['embeddings.LayerNorm.weight'].numel()  # LayerNorm层 gamma (1, 768)
    embeddings_LayerNorm_bias_para_cnt = state_dict['embeddings.LayerNorm.bias'].numel()  # LayerNorm层 beta (1, 768)
    embeddings_para_cnt = embeddings_word_embeddings_weight_para_cnt + \
                          embeddings_position_embeddings_weight_para_cnt + \
                          embeddings_token_type_embeddings_weight_para_cnt + \
                          embeddings_LayerNorm_weight_para_cnt + \
                          embeddings_LayerNorm_bias_para_cnt



    # embeddings 和 lastLiner 的key集合
    embeddings_and_lastLiner_keys_list = ['embeddings.word_embeddings.weight',
                                          'embeddings.position_embeddings.weight',
                                          'embeddings.token_type_embeddings.weight',
                                          'embeddings.LayerNorm.weight', 'embeddings.LayerNorm.bias',
                                          'pooler.dense.weight', 'pooler.dense.bias']

    # 计算transformers的参数量
    transformers_para_cnt = 0
    for key, value in state_dict.items():
        if key not in embeddings_and_lastLiner_keys_list:
            transformers_para_cnt += value.numel()

    # 最后一个线性层参数
    pooler_dense_weight_para_cnt = state_dict['pooler.dense.weight'].numel()  # W (768, 768)
    pooler_dense_bias_para_cnt = state_dict['pooler.dense.bias'].numel()  # b (1, 768)
    lastLiner_para_cnt = pooler_dense_weight_para_cnt + pooler_dense_bias_para_cnt

    # 汇总
    bert_base_all_para_cnt = embeddings_para_cnt + lastLiner_para_cnt + transformers_para_cnt * 12
    print("方法1结果= ", bert_base_all_para_cnt)


# 方法二：按步骤计算
def calcu_all_parameter_cnt2(vocab_size, hidden_size, num_layers):
    # 计算embeddings的参数量
    token_embedding_para_cnt = vocab_size * hidden_size  # token embedding (V, 768)
    segment_embedding_para_cnt = 2 * hidden_size  # segment embedding (2, 768)
    position_embedding_para_cnt = 512 * hidden_size  # position embedding (512, 768)
    embeddings_LayerNorm_weight_para_cnt = hidden_size * hidden_size  # embeddings后的LayerNorm层 gamma (1, 768)
    embeddings_LayerNorm_bias_para_cnt = 1 * hidden_size  # embeddings后的LayerNorm层 beta (1, 768)
    embeddings_para_cnt = token_embedding_para_cnt + \
                          segment_embedding_para_cnt + \
                          position_embedding_para_cnt + \
                          embeddings_LayerNorm_weight_para_cnt + \
                          embeddings_LayerNorm_bias_para_cnt

    # 计算multi-heads参数量（12 * 64 = 768 ，12头）
    # X(L,768)
    # Q_w(768,64) Q_b(1,64) --->Q(L,64)
    # K_w(768,64) K_b(1,64) --->K(L,64)
    # V_w(768,64) V_b(1,64) --->V(L,64)
    # Q*KT --->(L,L)
    # √dk=√64=8
    # Q * KT * V ---> （L,64）

    # 计算Q K V 三个线性层的参数
    encoder_layer_attention_self_query_weight_para_cnt = hidden_size * 64  # Q_w(768,64)
    encoder_layer_attention_self_query_bias_para_cnt = 1 * 64  # K_b(1,64)
    encoder_layer_attention_self_key_weight_para_cnt = hidden_size * 64  # K_w(768,64)
    encoder_layer_attention_self_key_bias_para_cnt = 1 * 64  # Q_b(1,64)
    encoder_layer_attention_self_value_weight_para_cnt = hidden_size * 64  # V_w(768,64)
    encoder_layer_attention_self_value_bias_para_cnt = 1 * 64  # V_b(1,64)

    # 𝑜𝑢𝑡𝑝𝑢𝑡 = 𝐿𝑖𝑛𝑒𝑟(𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛(𝑄, 𝐾, 𝑉))
    encoder_layer_attention_output_dense_weight_para_cnt = hidden_size * hidden_size  # 𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛 后的线性层W
    encoder_layer_attention_output_dense_bias_para_cnt = 1 * hidden_size  # 𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛 后的线性层b

    # 𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛 后的LayerNorm层 LayerNorm(Xembedding+ Xattention)
    encoder_layer_attention_output_LayerNorm_weight_para_cnt = 1 * hidden_size  # LayerNorm层 gamma (1, 768)
    encoder_layer_attention_output_LayerNorm_bias_para_cnt = 1 * hidden_size  # LayerNorm层 beta (1, 768)

    attention_para_cnt = (encoder_layer_attention_self_query_weight_para_cnt + \
                          encoder_layer_attention_self_query_bias_para_cnt + \
                          encoder_layer_attention_self_key_weight_para_cnt + \
                          encoder_layer_attention_self_key_bias_para_cnt + \
                          encoder_layer_attention_self_value_weight_para_cnt + \
                          encoder_layer_attention_self_value_bias_para_cnt) * 12 + \
                         encoder_layer_attention_output_dense_weight_para_cnt + \
                         encoder_layer_attention_output_dense_bias_para_cnt + \
                         encoder_layer_attention_output_LayerNorm_weight_para_cnt + \
                         encoder_layer_attention_output_LayerNorm_bias_para_cnt

    # 计算FeedForward层的参数
    # 𝑜𝑢𝑡𝑝𝑢𝑡 = 𝐿𝑖𝑛𝑒𝑟(𝑔𝑒𝑙𝑢(𝐿𝑖𝑛𝑒𝑟(𝑥)))
    # Bert沿用了惯用的全连接层大小设置，即4 * dmodle = 3072，因此，W1，W2分别为（768, 3072)，（3072, 768）
    encoder_layer_intermediate_dense_weight_para_cnt = 768 * 3702  # 里面的线性层W （768, 3072）
    encoder_layer_intermediate_dense_bias_para_cnt = 1 * 3702  # 里面的线性层b （1, 3702）
    encoder_layer_output_dense_weight_para_cnt = 3072 * 768  # 外面的线性层W （3072, 768）
    encoder_layer_output_dense_bias_para_cnt = 1 * 768  # 外面的线性层b （1, 3702）

    # FeedForward层后的LayerNorm层
    encoder_layer_output_LayerNorm_weight = 1 * hidden_size  # LayerNorm层 gamma (1, 768)
    encoder_layer_output_LayerNorm_bias = 1 * hidden_size  # LayerNorm层 beta (1, 768)

    feed_forward_para_cnt = encoder_layer_intermediate_dense_weight_para_cnt + \
                            encoder_layer_intermediate_dense_bias_para_cnt + \
                            encoder_layer_output_dense_weight_para_cnt + \
                            encoder_layer_output_dense_bias_para_cnt + \
                            encoder_layer_output_LayerNorm_weight + \
                            encoder_layer_output_LayerNorm_bias

    # 最后一个线性层参数
    pooler_dense_weight_para_cnt = hidden_size * hidden_size  # W (768, 768)
    pooler_dense_bias_para_cnt = 1 * hidden_size  # b (1, 768)
    lastLiner_para_cnt = pooler_dense_weight_para_cnt + pooler_dense_bias_para_cnt

    # 汇总
    bert_base_all_para_cnt = embeddings_para_cnt + (attention_para_cnt + feed_forward_para_cnt) * num_layers + lastLiner_para_cnt
    print("方法2结果= ", bert_base_all_para_cnt)


if __name__ == "__main__":
    calcu_all_parameter_cnt1()  # 102267648 ≈ 102 M ≈ 0.1B
    calcu_all_parameter_cnt2(21128, 768, 12)  # 108670344  ≈ 108 M ≈ 0.1B
