class CalculateBertParameterCount:
    def __init__(self, word_len, num_attention_heads=12, hidden_size=768, num_layers=1):
        """
        初始化MyClass的实例。

        参数:
            word_len (int): 词的个数 (< 512)。
            num_attention_heads (int): 多头机制中 ‘头’ 个数。
            hidden_size： 隐藏层 即 词向量的长度
            num_layers
        """
        self.word_len = word_len
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.num_layers = num_layers  # 注意这里的层数要跟预训练config.json文件中的模型层数一致

    def __calc_embedding_parameter_count(self):
        """
            return: embedding 层的参数量
        """
        em_count = self.word_len * self.hidden_size
        # segment_count = self.hidden_size * 2
        token_count = self.word_len * self.hidden_size
        position_count = self.word_len * self.hidden_size
        embeddings_layer_norm_weight_count = self.hidden_size #由于广播机制，这个向量实际只有 1 * hidden_size
        embeddings_layer_norm_bias_count = self.hidden_size  # 由于广播机制，这个向量实际只有 1 * hidden_size

        return sum([em_count,
                    token_count,
                    position_count,
                    embeddings_layer_norm_weight_count,
                    embeddings_layer_norm_bias_count])

    def __calc_attention_parameter_count(self):
        """
            return: 自注意层的参数量
        """
        qw_count = self.word_len * self.hidden_size
        kw_count = self.word_len * self.hidden_size
        vw_count = self.word_len * self.hidden_size

        qb_count = self.hidden_size #由于广播机制，这个向量实际只有 1 * hidden_size
        kb_count = self.hidden_size #由于广播机制，这个向量实际只有 1 * hidden_size
        vb_count = self.hidden_size #由于广播机制，这个向量实际只有 1 * hidden_size

        attention_output_weight_count = self.hidden_size * self.hidden_size
        attention_output_bias_count = self.hidden_size
        return sum([qw_count,
                    kw_count,
                    vw_count,
                    qb_count,
                    kb_count,
                    vb_count,
                    attention_output_weight_count,
                    attention_output_bias_count])

    def __calc_feed_forward_parameter_count(self):
        """

        :return: 前馈网络参数个数
        """
        intermediate_weight_count = (self.hidden_size * 4) * self.hidden_size
        intermediate_bias_count = self.hidden_size * 4

        output_weight_count = self.hidden_size * (self.hidden_size * 4)
        output_bias_count = self.hidden_size
        return sum([intermediate_weight_count, intermediate_bias_count, output_weight_count, output_bias_count])

    def calc_count(self):
        return (self.__calc_embedding_parameter_count()
                + self.num_layers * (
                        self.__calc_feed_forward_parameter_count()
                        + self.__calc_attention_parameter_count())
                )

if __name__ == '__main__':
    calc = CalculateBertParameterCount(word_len=4)
    print("参数量为", calc.calc_count())