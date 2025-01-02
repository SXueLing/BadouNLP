"""
...
def load_weights(self, state_dict):
    #embedding部分
    self.word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy() # shape: [vocab_size, hidden_size]
    self.position_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy() # shape: [max_len 512, hidden_size]
    self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy() # shape: [2, hidden_size]
    self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy() # shape: [hidden_size]
    self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy() # shape: [hidden_size]
    self.transformer_weights = [] # total parameters per layer: (hidden_size * hidden_size) * 4 + hidden_size * 9 + intermediate_size * hidden_size + intermediate_size
    
    #transformer部分，有多层
    for i in range(self.num_layers):
        q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % i].numpy() # shape: [hidden_size, hidden_size]
        q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % i].numpy() # shape: [hidden_size]
        k_w = state_dict["encoder.layer.%d.attention.self.key.weight" % i].numpy() # shape: [hidden_size, hidden_size]
        k_b = state_dict["encoder.layer.%d.attention.self.key.bias" % i].numpy() # shape: [hidden_size]
        v_w = state_dict["encoder.layer.%d.attention.self.value.weight" % i].numpy() # shape: [hidden_size, hidden_size]
        v_b = state_dict["encoder.layer.%d.attention.self.value.bias" % i].numpy() # shape: [hidden_size]
        attention_output_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % i].numpy() # shape: [hidden_size, hidden_size]
        attention_output_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % i].numpy() # shape: [hidden_size]
        attention_layer_norm_w = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i].numpy() # shape: [hidden_size]
        attention_layer_norm_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i].numpy() # shape: [hidden_size]
        intermediate_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % i].numpy() # shape: [hidden_size 768, intermediate_size 3072]
        intermediate_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % i].numpy() # shape: [intermediate_size 3072]
        output_weight = state_dict["encoder.layer.%d.output.dense.weight" % i].numpy() # shape: [intermediate_size 3072, hidden_size 768]
        output_bias = state_dict["encoder.layer.%d.output.dense.bias" % i].numpy() # shape: [hidden_size]
        ff_layer_norm_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i].numpy() # shape: [hidden_size]
        ff_layer_norm_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i].numpy() # shape: [hidden_size]
"""

"""
BERT total parameters (excluding pooler output layer)
= vocab_size * hidden_size + max_len * hidden_size + 2 * hidden_size + hidden_size * 2 + num_layers * [(hidden_size * hidden_size) * 4 + hidden_size * 9 + intermediate_size * hidden_size + intermediate_size]
= vocab_size * hidden_size + max_len * hidden_size + hidden_size * 4 + num_layers * ((hidden_size * hidden_size) * 4 + hidden_size * 9 + intermediate_size * hidden_size * 2 + intermediate_size)
"""

def count(num_layers, hidden_size, intermediate_size, max_len, vocab_size):
    word_embeddings = vocab_size * hidden_size
    position_embeddings = max_len * hidden_size
    token_type_embeddings = 2 * hidden_size
    embeddings_layer_norm_weight = hidden_size
    embeddings_layer_norm_bias = hidden_size
    embedding_weights = word_embeddings + position_embeddings + token_type_embeddings + embeddings_layer_norm_weight + embeddings_layer_norm_bias
    print(f"Embedding weights: {embedding_weights}")

    # total parameters per layer: (hidden_size * hidden_size) * 4 + hidden_size * 9 + intermediate_size * hidden_size + intermediate_size
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
    intermediate_weight = hidden_size * intermediate_size
    intermediate_bias = intermediate_size
    output_weight = intermediate_size * hidden_size
    output_bias = hidden_size
    ff_layer_norm_w = hidden_size
    ff_layer_norm_b = hidden_size    
    layer_weights = q_w + q_b + k_w + k_b + v_w + v_b + attention_output_weight + attention_output_bias + attention_layer_norm_w + attention_layer_norm_b + intermediate_weight + intermediate_bias + output_weight + output_bias + ff_layer_norm_w + ff_layer_norm_b
    print(f"Transformer weights (per layer): {layer_weights}")
    
    transformer_weights = embedding_weights + num_layers * layer_weights
    print(f"Total BERT weights: {transformer_weights}")

    return

if __name__ == "__main__":
    num_layers = 12
    hidden_size = 768
    intermediate_size = 3072
    max_len = 512
    vocab_size = 30522
    count(num_layers, hidden_size, intermediate_size, max_len, vocab_size)