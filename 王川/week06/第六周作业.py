"""
计算bert的参数总量
"""

def calc_bert_params(vocab_size, corpus_len = 512, vec_dim = 768, num_layers = 12):
    
    #embedding层参数总量
    token_embedding_nums = vocab_size * vec_dim
    segment_embedding_nums = 2 * vec_dim
    position_embedding_nums = corpus_len * vec_dim
    embedding_layer_norm_nums = 2 * vec_dim
    embedding_params_nums = token_embedding_nums + segment_embedding_nums + position_embedding_nums + embedding_layer_norm_nums

    #self-attention部分参数总量
    Qw_nums = vec_dim * vec_dim
    Qb_nums = vec_dim
    Kw_nums = vec_dim * vec_dim
    Kb_nums = vec_dim
    Vw_nums = vec_dim * vec_dim
    Vb_nums = vec_dim
    output_weight = vec_dim * vec_dim
    output_bias = vec_dim
    attention_layer_norm_nums = 2 * vec_dim
    attention_params_nums = Qw_nums + Qb_nums + Kw_nums + Kb_nums + Vw_nums + Vb_nums + output_weight + output_bias + attention_layer_norm_nums

    #feed_forward部分参数总量，假设第一层的hidden_size为3072
    feed_forward_linear1_weight = vec_dim * 3072
    feed_forward_linear1_bias = 3072
    feed_forward_linear2_weight = 3072 * vec_dim
    feed_forward_linear2_bias = vec_dim
    feed_forward_layer_norm_nums = 2 * vec_dim
    feed_forward_params_nums = feed_forward_layer_norm_nums + feed_forward_linear1_bias + feed_forward_linear1_weight + feed_forward_linear2_bias + feed_forward_linear2_weight

    #pooler部分参数
    pooler_dense_weight = vec_dim * vec_dim
    pooler_dense_bias = vec_dim
    pooler_params_nums = pooler_dense_weight + pooler_dense_bias

    return embedding_params_nums + num_layers * (feed_forward_params_nums + attention_params_nums) + pooler_params_nums


