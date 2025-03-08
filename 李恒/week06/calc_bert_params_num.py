def total_params(transformer_layers, num_attention_heads, hidden_size, vocab_size, word_length):
    def embedding_params_num():
        return (
            # token_embedding_size
            vocab_size * hidden_size +
            # segment_embedding_size
            hidden_size * 2 +
            # position_embedding_size
            word_length * hidden_size +
            # layer_norm_embedding_size w + b
            hidden_size * 2
        )

    def transformer_layer_params_num():
        return (
            # self_attention_params_num
            # query params
            hidden_size * word_length +
            # key params
            hidden_size * word_length +
            # value params
            hidden_size * word_length +
            # softmax params ?????
            hidden_size +

            # layer normalize params  w + b
            hidden_size * 2 +

            # feed forward params  linear + gelu + linear
            hidden_size * 4 * 2 +
            hidden_size * 2 +
            hidden_size * 2 +

            # layer normalize params  w + b
            hidden_size * 2

        )

    return embedding_params_num() + transformer_layer_params_num() * transformer_layers


if __name__ == '__main__':
    transformer_layers = 12
    num_attention_heads = 12
    hidden_size = 768
    vocab_size = 21128
    max_word_length = 512
    print(total_params(transformer_layers, num_attention_heads, hidden_size, vocab_size, max_word_length))