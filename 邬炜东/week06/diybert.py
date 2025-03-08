import torch
import numpy as np
from transformers import BertModel
import math
from transformers import BertTokenizer

Bert = BertModel.from_pretrained("bert-base-chinese/bert-base-chinese", return_dict=False)
state_dict = Bert.state_dict()
Bert.eval()
x = np.array([2450, 15486, 102, 2110])
print(state_dict.keys())


# 手写bert，num_layer为2

class BERT_DIY:
    def __init__(self, state_dict):
        self.num_layer = 2
        self.head_num = 12
        self.hidden_size = 768
        self.load_parameters(state_dict)

    # 加载参数
    def load_parameters(self, state_dict):
        self.embeddings_word_embeddings_weight = state_dict["embeddings.word_embeddings.weight"].numpy()
        self.embeddings_position_embeddings_weight = state_dict["embeddings.position_embeddings.weight"].numpy()
        self.embeddings_token_type_embeddings_weight = state_dict["embeddings.token_type_embeddings.weight"].numpy()
        self.embeddings_LayerNorm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
        self.embeddings_LayerNorm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()
        self.layers_parameters = []
        for i in range(self.num_layer):
            self.layers_parameters.append(
                [
                    state_dict["encoder.layer.%d.attention.self.query.weight" % i].numpy(),
                    state_dict["encoder.layer.%d.attention.self.query.bias" % i].numpy(),
                    state_dict["encoder.layer.%d.attention.self.key.weight" % i].numpy(),
                    state_dict["encoder.layer.%d.attention.self.key.bias" % i].numpy(),
                    state_dict["encoder.layer.%d.attention.self.value.weight" % i].numpy(),
                    state_dict["encoder.layer.%d.attention.self.value.bias" % i].numpy(),
                    state_dict["encoder.layer.%d.attention.output.dense.weight" % i].numpy(),
                    state_dict["encoder.layer.%d.attention.output.dense.bias" % i].numpy(),
                    state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i].numpy(),
                    state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i].numpy(),
                    state_dict["encoder.layer.%d.intermediate.dense.weight" % i].numpy(),
                    state_dict["encoder.layer.%d.intermediate.dense.bias" % i].numpy(),
                    state_dict["encoder.layer.%d.output.dense.weight" % i].numpy(),
                    state_dict["encoder.layer.%d.output.dense.bias" % i].numpy(),
                    state_dict["encoder.layer.%d.output.LayerNorm.weight" % i].numpy(),
                    state_dict["encoder.layer.%d.output.LayerNorm.bias" % i].numpy()
                ]
            )

    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    # 归一化层
    def Layer_Normalizatoin(self, x, w, b):
        x = (x - np.mean(x, axis=1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + 1e-12)
        return x * w + b

    # 多头切分
    def to_multi_head(self, x, one_head_dim):  # (4,768)
        seq_len, hidden_layer = x.shape
        x = x.reshape(seq_len, self.head_num, one_head_dim)  # (4,12,64)
        return x.swapaxes(0, 1)  # (12, 4, 64)

    # 多头计算
    def multi_head_calculate(self, Q, K, V, one_head_dim):  # (12, 4, 64)
        QK = self.softmax(np.matmul(Q, K.swapaxes(1, 2)) / np.sqrt(one_head_dim))  # (12, 4, 4)
        QKV = np.matmul(QK, V)  # (12, 4, 64)
        return QKV

    # 加载嵌入
    def get_embedding(self, x):
        we = self.embeddings_word_embeddings_weight
        pe = self.embeddings_position_embeddings_weight
        te = self.embeddings_token_type_embeddings_weight
        x_word = x
        x_position = list(range(len(x)))
        x_type = [0] * len(x)
        word_embedding = np.array([we[index] for index in x_word])
        position_embedding = np.array([pe[index] for index in x_position])
        type_embedding = np.array([te[index] for index in x_type])
        total_embedding = word_embedding + position_embedding + type_embedding
        x = self.Layer_Normalizatoin(total_embedding, self.embeddings_LayerNorm_weight, self.embeddings_LayerNorm_bias)
        return x

    def all_layers(self, x):
        for i in range(self.num_layer):
            x = self.one_layer(x, i)
        return x

    # 全连接层的运算
    def matrix_dot(self, x, w, b):
        return np.dot(x, w.T) + b

    def one_layer(self, x, layer):
        q_w, q_b, k_w, k_b, v_w, v_b, ao_w, ao_b, aol_w, aol_b, ff_w, ff_b, o_w, o_b, ol_w, ol_b = \
            self.layers_parameters[layer]
        Q = self.matrix_dot(x, q_w, q_b)
        K = self.matrix_dot(x, k_w, k_b)
        V = self.matrix_dot(x, v_w, v_b)  # (4, 768)
        one_head_dim = int(self.hidden_size / self.head_num)
        Q = self.to_multi_head(Q, one_head_dim)  # (12, 4, 64)
        K = self.to_multi_head(K, one_head_dim)
        V = self.to_multi_head(V, one_head_dim)
        QKV = self.multi_head_calculate(Q, K, V, one_head_dim)  # (12, 4, 64)
        QKV = QKV.swapaxes(0, 1).reshape(-1, self.hidden_size)  # (4, 12, 64)(4, 768)
        attention_x = self.matrix_dot(QKV, ao_w, ao_b)  # (4, 768)
        x = self.Layer_Normalizatoin(x + attention_x, aol_w, aol_b)  # (4, 768)
        ff_x = self.gelu(self.matrix_dot(x, ff_w, ff_b))  # (4, 3072)
        ff_x = self.matrix_dot(ff_x, o_w, o_b)  # (4, 768)
        x = self.Layer_Normalizatoin(x + ff_x, ol_w, ol_b)
        return x

    def get_pool(self, x):
        x = self.matrix_dot(x, self.pooler_dense_weight, self.pooler_dense_bias)
        x = np.tanh(x)
        return x

    def main(self, x):
        x = self.get_embedding(x)
        y_bert_diy = self.all_layers(x)
        y_bert_pool = self.get_pool(y_bert_diy[0])
        return y_bert_diy, y_bert_pool


Bert_diy = BERT_DIY(state_dict)
y_bert_diy, y_bert_diy_pool = Bert_diy.main(x)
x_input = torch.LongTensor([x])
y_bert, y_bert_pool = Bert(x_input)
print(y_bert_diy)
print(y_bert)


# 计算参数量
def count_parameters_from_state_dict(state_dict):
    total = 0
    for name, parameter in state_dict.items():
        print(name)
        print(parameter.shape)
        print(parameter.numel())
        total += parameter.numel()
        print("==========================================")
    return total


total = count_parameters_from_state_dict(state_dict)
print(f"Total number of parameters: {total}")
