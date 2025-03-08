# debug看到的参数形状
"""
    def load_weights(self, state_dict):
        #embedding部分
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy() #[vocab 21128,hidden_size 768]
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()  #[max_len 512,hidden_size 768]
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()  #[2,hidden_size 768]
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()  #[hidden_size 768]
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy() #[hidden_size 768]
        self.transformer_weights = []
        #transformer部分，有多层
        for i in range(self.num_layers):
            q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % i].numpy() #[hidden_size 768,hidden_size 768]
            q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % i].numpy() #[hidden_size 768]
            k_w = state_dict["encoder.layer.%d.attention.self.key.weight" % i].numpy() #[hidden_size 768,hidden_size 768]
            k_b = state_dict["encoder.layer.%d.attention.self.key.bias" % i].numpy()  #[hidden_size 768]
            v_w = state_dict["encoder.layer.%d.attention.self.value.weight" % i].numpy()  #[hidden_size 768,hidden_size 768]
            v_b = state_dict["encoder.layer.%d.attention.self.value.bias" % i].numpy()  #[hidden_size 768]
            attention_output_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % i].numpy()  #[hidden_size 768,hidden_size 768]
            attention_output_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % i].numpy() #[hidden_size 768]
            attention_layer_norm_w = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i].numpy()  #[hidden_size 768]
            attention_layer_norm_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i].numpy()  #[hidden_size 768]
            intermediate_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % i].numpy()  #[intermediate_size 3072,hidden_size 768] 转至得换位置
            intermediate_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % i].numpy() #[intermediate_size 3072]
            output_weight = state_dict["encoder.layer.%d.output.dense.weight" % i].numpy()  #[hidden_size 768,intermediate_size 3072]  转至得换位置
            output_bias = state_dict["encoder.layer.%d.output.dense.bias" % i].numpy()  #[hidden_size 768]
            ff_layer_norm_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i].numpy()  #[hidden_size 768]
            ff_layer_norm_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i].numpy()  #[hidden_size 768]
            self.transformer_weights.append([q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                                             attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias,
                                             output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])
        #pooler层
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy() #[hidden_size 768,hidden_size 768]
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy() #[hidden_size 768]
"""


def bert_sum(num_layers, hidden_size, intermediate_size, max_len, vocab_size):
    word_embeddings = vocab_size * hidden_size
    position_embeddings = max_len * hidden_size
    token_type_embeddings = 2 * hidden_size
    embeddings_layer_norm_weight = hidden_size
    embeddings_layer_norm_bias = hidden_size
    embeddings_weights = word_embeddings + position_embeddings + token_type_embeddings + embeddings_layer_norm_weight + embeddings_layer_norm_bias
    print('Embedddings weights:', embeddings_weights)

    q_w = hidden_size * hidden_size
    q_b = hidden_size
    k_w = hidden_size * hidden_size
    k_b = hidden_size
    v_w = hidden_size * hidden_size
    v_b = hidden_size
    attention_output_weight = hidden_size * hidden_size
    attention_output_bias = hidden_size
    attention_layer_norm_w = hidden_size
    attention_layer_norm_b = hidden_size
    intermediate_weight = intermediate_size * hidden_size
    intermediate_bias = intermediate_size
    output_weight = hidden_size * intermediate_size
    output_bias = hidden_size
    ff_layer_norm_w = hidden_size
    ff_layer_norm_b = hidden_size
    layer_weight = q_w + q_b + k_w + k_b + v_w + v_b + attention_output_weight + attention_output_bias \
                   + attention_layer_norm_w + attention_layer_norm_b + intermediate_weight + intermediate_bias \
                   + output_weight + output_bias + ff_layer_norm_w + ff_layer_norm_b
    print('transformer部分 weights:', layer_weight)

    bert_weight = embeddings_weights + layer_weight * num_layers
    print('bert_weight:', bert_weight)


num_layers = 1
hidden_size = 768
intermediate_size = 3072
max_len = 512
vocab_size = 21128
bert_sum(num_layers, hidden_size, intermediate_size, max_len, vocab_size)
