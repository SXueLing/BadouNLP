# 计算 Bert 参数量

def count_parameters(vocab_size, word_embedding_size, len_max, hidden_size, ffn_size, num_transformer_layer, num_head = None):
    # embedding 层
    word_embedding = vocab_size * word_embedding_size
    sentence_embedding = 2 * word_embedding_size
    position_embedding = len_max * word_embedding_size

    embedding_parameters = word_embedding + sentence_embedding + position_embedding

    # 多头
    # num_head = num_head

    # layer_norm
    wln = hidden_size
    bln = hidden_size

    # self attention
    wq = word_embedding_size * hidden_size
    wk = wq
    wv = wq
    bq = hidden_size
    bk = bq
    bv = bq

    wo = hidden_size * hidden_size
    bo = hidden_size

    attention_parameters = wq + wk + wv + bq + bk + bv + wo + bo + wln + bln

    # FFN
    w1 = hidden_size * ffn_size
    b1 = ffn_size
    w2 = ffn_size * hidden_size
    b2 = hidden_size

    ffn_parameters = w1 + b1 + w2 + b2 + wln + bln

    # num_transformer_layer
    num_transformer_layer = num_transformer_layer

    return embedding_parameters + num_transformer_layer * (attention_parameters + ffn_parameters)

