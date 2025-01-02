
def cal_bert_params(num_transformer, vocab_size, pooler_fc_size,max_position_embeddings,intermediate_size):
    def cal_embedding_params(vocab_size, pooler_fc_size, max_position_embeddings):
        # 计算Embedding层
        token_embedding = vocab_size * pooler_fc_size
        segment_embedding = 2 * pooler_fc_size
        position_embedding = max_position_embeddings * pooler_fc_size
        # Embedding最后归一化
        embedding_norm_w,embedding_norm_b = pooler_fc_size,pooler_fc_size
        return token_embedding + segment_embedding + position_embedding + embedding_norm_w + embedding_norm_b

    def cal_single_transformer_params(pooler_fc_size, intermediate_size):
        # Self-Attention层
        q_w = pooler_fc_size * pooler_fc_size
        q_b = pooler_fc_size
        k_w = pooler_fc_size * pooler_fc_size
        k_b = pooler_fc_size
        v_w = pooler_fc_size * pooler_fc_size
        v_b = pooler_fc_size
        # linear层
        linear_attention_output_weight = pooler_fc_size * pooler_fc_size
        linear_attention_output_bias = pooler_fc_size
        # layernorm
        layer_norm_w = pooler_fc_size
        layer_norm_b = pooler_fc_size
        # feedforward层
        feedforward_fc1 = pooler_fc_size * intermediate_size
        feedforward_fc1_bias = intermediate_size
        feedforward_fc2 = intermediate_size * pooler_fc_size
        feedforward_fc2_bias = pooler_fc_size
        # layernorm
        feedforward_layer_norm_w = pooler_fc_size
        feedforward_layer_norm_b = pooler_fc_size
        return q_w + q_b + k_w + k_b + v_w + v_b + linear_attention_output_weight + linear_attention_output_bias + layer_norm_w + layer_norm_b + feedforward_fc1 + feedforward_fc1_bias + feedforward_fc2 + feedforward_fc2_bias + feedforward_layer_norm_w + feedforward_layer_norm_b
    
    def cal_pooler_params(pooler_fc_size):
        # Pooler层
        pooler_fc = pooler_fc_size * pooler_fc_size
        pooler_fc_bias = pooler_fc_size
        return pooler_fc + pooler_fc_bias

    return cal_embedding_params(vocab_size, pooler_fc_size, max_position_embeddings) + num_transformer*cal_single_transformer_params(pooler_fc_size, intermediate_size) + cal_pooler_params(pooler_fc_size)


# transformer 层数
num_transformer = 12
# 词表大小 config.json中获取
vocab_size = 21128
# 词向量维度
pooler_fc_size = 768
# 最大位置编码
max_position_embeddings = 512
# feedforward层维度
intermediate_size = 3072


bert_params = cal_bert_params(num_transformer, vocab_size, pooler_fc_size,max_position_embeddings,intermediate_size)
print(bert_params)

