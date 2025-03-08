
MODEL_NAME = "bert-base-chinese"

def paramters_statistic(vocab_size=21128, hidden_size=768, transformer_layersv=12):
    """
    函数用来计算该模型用于训练的参数数量

    根据模型的配置文件使用以下参数的默认值
    inargs:
    词表大小 ： vocab_size=21128
    词表的向量维度： hidden_size=768
    transformer层数： transformer_layersv=12
    outarg:
    return 需要训练的参数两
    """
    total_parameters = 0

    #input embedding layer
    max_position_embeddings = 512
    para_word_embeddings = vocab_size * hidden_size
    para_position_embeddings = max_position_embeddings * hidden_size
    para_token_embeddings = 2 * hidden_size # 一共两个token,开始标记[CLS]和结束标记[SEP]
    para_input_embeddings_norm = 2 * hidden_size # w:768, b:768
    total_parameters += para_word_embeddings + para_position_embeddings + para_token_embeddings + para_input_embeddings_norm

    # transformer layer
    ## self-attention layer
    total_transformer_params = 0
    Qw = hidden_size * hidden_size
    Qb = hidden_size
    Kw = hidden_size * hidden_size
    Kb = hidden_size
    Vw = hidden_size * hidden_size
    Vb = hidden_size
    attention_linear_w = hidden_size * hidden_size
    attention_linear_b = hidden_size
    para_attention_output = Qw + Qb + Kw + Kb + Vw + Vb + attention_linear_w + attention_linear_b
    total_transformer_params += para_attention_output

    ## self-attention output normlization
    param_attention_output_norm = 2 * hidden_size # w:768 b:768
    total_transformer_params += param_attention_output_norm

    ## feed forward layer
    intermediate_size = 3072
    param_feedforward_linear1_w = hidden_size * intermediate_size
    param_feedforward_linear1_b = intermediate_size
    param_feedforward_linear2_w = intermediate_size * hidden_size
    param_feedforward_linear2_b = hidden_size
    total_transformer_params += param_feedforward_linear2_w + param_feedforward_linear2_b + param_feedforward_linear1_w + param_feedforward_linear1_b

    ##feed forward layer norm
    param_feedforward_norm = 2 * hidden_size # w:768 b:768
    total_transformer_params += param_feedforward_norm
    total_parameters += total_transformer_params * transformer_layersv

    #pooler layer
    param_pooler_linear_w = hidden_size * hidden_size
    param_pooler_linear_b = hidden_size
    total_parameters += param_pooler_linear_w + param_pooler_linear_b

    return total_parameters

num_params = paramters_statistic()
print(f"{MODEL_NAME} model parameters should be trained: {num_params}")
