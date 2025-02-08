# 尝试自定义自注意力机制和前馈
import numpy as np
from transformers import BertModel
import torch


def soft_max(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def ge_lu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt((2 / np.pi)) * (x + 0.044715 * x ** 3)))


class CustomSelfAttentionModel:
    def __init__(self, state_dict, transformer_count=1, hidden_size=768):
        self.num_header = 12
        self.transformer_count = transformer_count
        self.hidden_size = hidden_size
        self.load_weights(state_dict)

    def get_weight(self, head):

        pass

    def load_weights(self, state_dict):
        all_param = 0
        # 拿到embedding的参数
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy()
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()
        self.transformer_weights = []
        # 拿到transformer的参数
        for i in range(self.transformer_count):
            q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % i].numpy()
            q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % i].numpy()
            k_w = state_dict["encoder.layer.%d.attention.self.key.weight" % i].numpy()
            k_b = state_dict["encoder.layer.%d.attention.self.key.bias" % i].numpy()
            v_w = state_dict["encoder.layer.%d.attention.self.value.weight" % i].numpy()
            v_b = state_dict["encoder.layer.%d.attention.self.value.bias" % i].numpy()
            attention_output_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % i].numpy()
            attention_output_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % i].numpy()
            attention_layer_norm_w = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i].numpy()
            attention_layer_norm_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i].numpy()
            intermediate_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % i].numpy()
            intermediate_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % i].numpy()
            output_weight = state_dict["encoder.layer.%d.output.dense.weight" % i].numpy()
            output_bias = state_dict["encoder.layer.%d.output.dense.bias" % i].numpy()
            ff_layer_norm_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i].numpy()
            ff_layer_norm_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i].numpy()
            self.transformer_weights.append(
                [q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                 attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias,
                 output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])

        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()

    # embedding层实际上相当于按index索引，或理解为onehot输入乘以embedding矩阵
    def get_embedding(self, embedding_matrix, x):
        return np.array([embedding_matrix[index] for index in x])

    def all_calc(self, x, transformer_weights):
        weight_count = 0
        # 获取所有的embedding
        x, embedding_weight_count = self.calc_embedding_all(x)
        weight_count += embedding_weight_count
        # 计算每一个single 的transformer
        for i in range(self.transformer_count):
            (q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias, attention_layer_norm_w,
             attention_layer_norm_b, intermediate_weight, intermediate_bias, output_weight, output_bias,
             ff_layer_norm_w,
             ff_layer_norm_b) = transformer_weights[i]

            calc_x, single_weight = self.calc_single_transformer(x, q_w, q_b, k_w, k_b, v_w, v_b,
                                                                 attention_output_weight,
                                                                 attention_output_bias)
            weight_count += single_weight
            x = self.layer_norm((x + calc_x), attention_layer_norm_w, attention_layer_norm_b)
            weight_count += torch.numel(torch.tensor(attention_layer_norm_w))
            weight_count += torch.numel(torch.tensor(attention_layer_norm_b))
            forward_x = self.calc_ff(x, intermediate_weight, intermediate_bias, output_weight, output_bias)
            weight_count += torch.numel(torch.tensor(output_weight))
            weight_count += torch.numel(torch.tensor(output_bias))
            x = self.layer_norm(x + forward_x, ff_layer_norm_w, ff_layer_norm_b)
            weight_count += torch.numel(torch.tensor(ff_layer_norm_w))
            weight_count += torch.numel(torch.tensor(ff_layer_norm_b))
        # 过ff的计算
        return x, weight_count

    def calc_ff(self, x, intermediate_weight, intermediate_bias, output_weight, output_bias):  # x = max_len , 768
        x = np.dot(x, intermediate_weight.T) + intermediate_bias
        x = ge_lu(x)
        x = np.dot(x, output_weight.T) + output_bias
        return x

        # 多头机制

    def transpose_for_scores(self, x, attention_head_size, num_attention_heads):
        # hidden_size = 768  num_attent_heads = 12 attention_head_size = 64
        max_len, hidden_size = x.shape
        x = x.reshape(max_len, num_attention_heads, attention_head_size)
        x = x.swapaxes(1, 0)  # output shape = [num_attention_heads, max_len, attention_head_size]
        return x

    def calc_single_transformer(self, x, q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight,
                                attention_output_bias):
        single_weight = 0
        single_weight += torch.numel(torch.tensor(q_w))
        single_weight += torch.numel(torch.tensor(q_b))
        single_weight += torch.numel(torch.tensor(k_w))
        single_weight += torch.numel(torch.tensor(k_b))
        single_weight += torch.numel(torch.tensor(v_w))
        single_weight += torch.numel(torch.tensor(v_b))
        single_weight += torch.numel(torch.tensor(attention_output_weight))
        single_weight += torch.numel(torch.tensor(attention_output_bias))
        # 算q q.shape: [max_len, 768]
        q = np.dot(x, q_w.T) + q_b.T
        # 算k k.shape: [max_len, 768]
        k = np.dot(x, k_w.T) + k_b.T
        # 算v v.shape: [max_len, 768]
        v = np.dot(x, v_w.T) + v_b.T
        max_len, hidden_size = x.shape
        # 多头计算
        attention_head_size = int(hidden_size / self.num_header)
        # 计算q多头以后的数据 q.shape = num_attention_heads, max_len, attention_head_size
        q = self.transpose_for_scores(q, attention_head_size, self.num_header)
        # 计算k多头以后的数据  k.shape = num_attention_heads, max_len, attention_head_size
        k = self.transpose_for_scores(k, attention_head_size, self.num_header)
        # 计算v多头以后的数据 v.shape = num_attention_heads, max_len, attention_head_size
        v = self.transpose_for_scores(v, attention_head_size, self.num_header)
        # qk.shape = num_attention_heads, max_len, max_len
        qk = np.matmul(q, k.swapaxes(1, 2))
        qk /= np.sqrt(attention_head_size)
        qk = soft_max(qk)
        # qkv.shape = num_attention_heads, max_len, attention_head_size
        qkv = np.matmul(qk, v)
        # qkv.shape = max_len, hidden_size
        qkv = qkv.swapaxes(0, 1).reshape(-1, hidden_size)
        # attention.shape = max_len, hidden_size  最后过一次线性层
        attention = np.dot(qkv, attention_output_weight.T) + attention_output_bias
        return attention, single_weight

    # 层归一化层
    def layer_norm(self, x, w, b):  # x => len * 768
        avg = np.mean(x, axis=1, keepdims=True)  # 平均数
        std = np.std(x, axis=1, keepdims=True)  # 标准差
        # 归一化
        one_data = (x - avg) / (std + 1e-5)
        return (w * one_data + b)

    def calc_embedding_all(self, x):
        weights_count = 0
        # 第一次
        token_embedding = self.get_embedding(self.word_embeddings, x)  # token_embedding = [len * 768]
        weights_count += torch.numel(torch.tensor(self.word_embeddings))
        position_arr = np.array(list(range(len(x))))
        seg_arr = np.array([0] * len(x))
        segment_embedding = self.get_embedding(self.position_embeddings, position_arr)  # token_embedding =[len * 768]
        weights_count += torch.numel(torch.tensor(self.position_embeddings))
        position_embedding = self.get_embedding(self.token_type_embeddings, seg_arr)  # position_embedding =[len * 768]
        weights_count += torch.numel(torch.tensor(self.token_type_embeddings))
        final_embedding = token_embedding + segment_embedding + position_embedding
        # layer层归一化
        embedding = self.layer_norm(final_embedding, self.embeddings_layer_norm_weight, self.embeddings_layer_norm_bias)
        weights_count += torch.numel(torch.tensor(self.embeddings_layer_norm_weight))
        weights_count += torch.numel(torch.tensor(self.embeddings_layer_norm_bias))
        return embedding, weights_count

    def forward(self, x):
        x, weight_count = self.all_calc(x, self.transformer_weights)
        pooler_output = self.pooler_output_layer(x[0])
        weight_count += torch.numel(torch.tensor(self.pooler_dense_weight))
        weight_count += torch.numel(torch.tensor(self.pooler_dense_bias))
        return x, pooler_output, weight_count

    def pooler_output_layer(self, x):
        x = np.dot(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
        x = np.tanh(x)
        return x


if __name__ == '__main__':
    bert = BertModel.from_pretrained(r"D:\JianGuo\AI\八斗\课件资料\第六周 语言模型\bert-base-chinese",
                                     return_dict=False)
    google_bert_state_dict = bert.state_dict()
    # 先计算一个transformer的结构
    # 3个embedding层
    x = np.array([2450, 15486, 102, 2111])
    # bert.eval()
    # torch_x = torch.LongTensor([x])
    # google_result, google_pooler_output = bert.forward(torch_x)
    # print(google_result)
    # print('===========================================')
    transformer_count = 1
    hidden_size = 768
    custom_bert = CustomSelfAttentionModel(google_bert_state_dict, transformer_count, hidden_size)
    dic_result, pooler_output, weight_count = custom_bert.forward(x)
    print(dic_result)
    print('=============================================')
    print('一共有%s个参数,大概等于%sB的参数' % (weight_count, np.round(weight_count / np.int64(10 ** 8), 3)))
