def params_size(num_layer, vocab_size):  # 参数个数计算
    num_layer = num_layer  # 中间层的数量
    hidden_size = 768  # 隐藏层大小 bert默认大小
    vocab_size = vocab_size  # 词表大小
    position_embeddings_length = 512  # 最大词数量， 最大支撑长度, position_embennding层最大的位置数量大小, bert默认大小， 绝对位置编码
    # embedding 参数大小 = token_embedding + segment_embedding + position_embennding + 一个layer_noram
    # 通过 embedding 之后会生成一个 文本长度 * 768 的向量
    embedding_params_num = vocab_size * hidden_size + 2 * hidden_size + position_embeddings_length * hidden_size + 2 * hidden_size
    # self attention 层参数大小 = k q v 三个 线性层 + 一个输出线性层 + 一个layer_noram
    attention_params_num = 4 * (hidden_size * hidden_size + hidden_size) + (2 * hidden_size)
    # feed forward 层参数 两个线性层 + 一个归一化层
    feed_forward_params_num = 2 * (hidden_size * hidden_size + hidden_size) + (2 * hidden_size)
    # pooler 层参数
    pooler_num = hidden_size * hidden_size + hidden_size
    # 总参数个数
    sum_num = embedding_params_num + num_layer * (attention_params_num + feed_forward_params_num) + pooler_num

    return sum_num
