import numpy as np


def GeLU(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x):
    """
    BNN
    """
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

class Module:
    def forward(self, x):
        pass

    def __call__(self, x):
        return self.forward(x)

class EncoderLayer(Module):
    def __init__(
        self,
        attention,
        attn_ln,
        intermediate,
        output,
        ln,
    ):
        """
        attn是MHA
        ln是层归一化单元
        intermediate和output都是最后的MLP的一部分
        """
        self.attn = attention
        self.attn_ln = attn_ln
        self.ln = ln
        self.intermediate = intermediate
        self.output = output

    def forward(self, x):
        attention_x = self.attn(x)
        attention_x += x # Residual
        intermediate_x = self.attn_ln(attention_x)
        raw_output = self.intermediate(intermediate_x)
        # 这里因为被state_dict误导了，把output设置成了一个单独的带ln的模块
        return self.ln(intermediate_x + self.output(GeLU(raw_output)))

class Encoder(Module):
    def __init__(self, encoder_layers):
        self.encoder_layers = encoder_layers

    def forward(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return x

class Linear(Module):
    """
    本质上，所有的层几乎都是线性层
    """
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, x):
        return np.dot(x, self.weights.T) + self.bias

class LayerNorm(Module):
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    """
    LN也包含了线性层，只需要重写forward即可
    """
    def forward(self, x):
        miu = np.mean(x, axis=-1, keepdims=True) # ND
        std = np.std(x, axis=-1, keepdims=True) # ND
        x = (x - miu) / std
        return x * self.weights + self.bias

class Attention(Module):
    def __init__(
        self,
        query,
        key,
        value,
        output,
        hidden_size,
        num_head,
    ):
        self.query = query
        self.key = key
        self.value = value
        self.output = output
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.head_dimension = self.hidden_size // self.num_head

    def forward(self, x):
        """
        x (N, D)
        """
        N, D = x.shape
        # 过线性层
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # 分头
        query = query.reshape(N, self.num_head, self.head_dimension) # N H DK
        key = key.reshape(N, self.num_head, self.head_dimension)
        value = value.reshape(N, self.num_head, self.head_dimension)
        query = np.swapaxes(query, 0, 1)  # H N DK
        key = np.swapaxes(key, 0, 1)
        value = np.swapaxes(value, 0, 1)

        # 计算注意力分数
        # H N DK * H DK N -> H N N
        attention_weight = softmax(np.matmul(query, np.swapaxes(key, 1, 2)) / np.sqrt(self.head_dimension))
        result = np.matmul(attention_weight, value) # H N N * H N DK -> H N DK
        result = np.swapaxes(result, 0, 1).reshape(N, D) # N H DK
        return self.output(result)

class Intermediate(Linear):
    pass

class Pooler(Module):
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        return np.tanh(np.dot(x, self.weight.T + self.bias))

class Embedding(Module):
    def __init__(self, word_embedding, positional_embedding, token_embedding):
        self.word_embedding = word_embedding # V D
        self.positional_embedding = positional_embedding
        self.token_embedding = token_embedding

    def forward(self, x):
        """
        这里x是一个N的单维矩阵
        """
        n = len(x)
        x = self.word_embedding[x] # N D
        segment_embedding = self.token_embedding[[0] * n] # 单句话直接全0向量
        position_embedding = self.positional_embedding[np.arange(n)] # 简单的暴力Positional Embedding
        return x + segment_embedding + position_embedding

class BERT(Module):
    def __init__(
        self,
        state_dict,
    ):
        self.weight_dict = state_dict
        self.num_layers = 1
        self.callable_modules = []
        self.load_weights()

    def _init_embedding(
        self,
        word_embed_wt,
        token_embed_wt,
        pos_embed_wt,
        ln_weight,
        ln_bia,
    ):
        self.embedding = Embedding(word_embed_wt, pos_embed_wt, token_embed_wt)
        self.embedding_ln = LayerNorm(ln_weight, ln_bia)
        self.callable_modules += [
            self.embedding, self.embedding_ln
        ]

    def _init_encoder_layer(
        self,
        query_wt,
        query_bias,
        key_wt,
        key_bias,
        value_wt,
        value_bias,
        attn_output_wt,
        attn_output_bias,
        attn_output_ln_wt,
        attn_output_ln_bias,
        intermediate_wt,
        intermediate_bias,
        output_wt,
        output_bias,
        output_ln_wt,
        output_ln_bias,
    ):
        attn = Attention(
            Linear(query_wt, query_bias),
            Linear(key_wt, key_bias),
            Linear(value_wt, value_bias),
            Linear(attn_output_wt, attn_output_bias),
            768,
            12
        )
        attn_ln = LayerNorm(attn_output_ln_wt, attn_output_ln_bias)
        intermediate = Intermediate(intermediate_wt, intermediate_bias)
        output = Linear(output_wt, output_bias)
        ln = LayerNorm(output_ln_wt, output_ln_bias)
        return EncoderLayer(
            attn,
            attn_ln,
            intermediate,
            output,
            ln
        )

    def _init_encoder(self, layers):
        self.encoder = Encoder(layers)
        self.callable_modules.append(self.encoder)

    def load_weights(self):
        self.word_embedding_weights = self.weight_dict["embeddings.word_embeddings.weight"].numpy()
        self.token_type_embedding_weights = self.weight_dict["embeddings.token_type_embeddings.weight"].numpy()
        self.position_embedding_weights = self.weight_dict["embeddings.position_embeddings.weight"].numpy()
        self.ln_weight = self.weight_dict["embeddings.LayerNorm.weight"].numpy()
        self.ln_bia = self.weight_dict["embeddings.LayerNorm.bias"].numpy()
        self._init_embedding(
            self.word_embedding_weights,
            self.token_type_embedding_weights,
            self.position_embedding_weights,
            self.ln_weight,
            self.ln_bia
        )
        layers = []
        for i in range(self.num_layers):
            layers.append(self._init_encoder_layer(
                self.weight_dict[f"encoder.layer.{i}.attention.self.query.weight"].numpy(),
                self.weight_dict[f"encoder.layer.{i}.attention.self.query.bias"].numpy(),
                self.weight_dict[f"encoder.layer.{i}.attention.self.key.weight"].numpy(),
                self.weight_dict[f"encoder.layer.{i}.attention.self.key.bias"].numpy(),
                self.weight_dict[f"encoder.layer.{i}.attention.self.value.weight"].numpy(),
                self.weight_dict[f"encoder.layer.{i}.attention.self.value.bias"].numpy(),
                self.weight_dict[f"encoder.layer.{i}.attention.output.dense.weight"].numpy(),
                self.weight_dict[f"encoder.layer.{i}.attention.output.dense.bias"].numpy(),
                self.weight_dict[f"encoder.layer.{i}.attention.output.LayerNorm.weight"].numpy(),
                self.weight_dict[f"encoder.layer.{i}.attention.output.LayerNorm.bias"].numpy(),
                self.weight_dict[f"encoder.layer.{i}.intermediate.dense.weight"].numpy(),
                self.weight_dict[f"encoder.layer.{i}.intermediate.dense.bias"].numpy(),
                self.weight_dict[f"encoder.layer.{i}.output.dense.weight"].numpy(),
                self.weight_dict[f"encoder.layer.{i}.output.dense.bias"].numpy(),
                self.weight_dict[f"encoder.layer.{i}.output.LayerNorm.weight"].numpy(),
                self.weight_dict[f"encoder.layer.{i}.output.LayerNorm.bias"].numpy()
            ))
        self._init_encoder(layers)
        self.pooler = Pooler(
            self.weight_dict["pooler.dense.weight"],
            self.weight_dict["pooler.dense.bias"]
        )

        """
        encoders weight
        'encoder.layer.0.attention.self.query.weight', 
        'encoder.layer.0.attention.self.query.bias', 
        'encoder.layer.0.attention.self.key.weight', 
        'encoder.layer.0.attention.self.key.bias', 
        'encoder.layer.0.attention.self.value.weight', 
        'encoder.layer.0.attention.self.value.bias', 
        'encoder.layer.0.attention.output.dense.weight', 
        'encoder.layer.0.attention.output.dense.bias', 
        'encoder.layer.0.attention.output.LayerNorm.weight', 
        'encoder.layer.0.attention.output.LayerNorm.bias', 
        'encoder.layer.0.intermediate.dense.weight', 
        'encoder.layer.0.intermediate.dense.bias', 
        'encoder.layer.0.output.dense.weight', 
        'encoder.layer.0.output.dense.bias', 
        'encoder.layer.0.output.LayerNorm.weight',
        'encoder.layer.0.output.LayerNorm.bias'
        """

    def forward(self, x):
        for index, func in enumerate(self.callable_modules):
            x = func(x)
        return x, self.pooler(x[0])
