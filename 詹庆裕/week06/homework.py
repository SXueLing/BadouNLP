import torch
import math
import numpy as np
from transformers import BertModel

bert = BertModel.from_pretrained(r"D:\NLP\bert-base-chinese", return_dict=False)
state_dicts = bert.state_dict()
bert.eval()
x = np.array([2450, 15486, 102, 2110]) 
torch_x = torch.LongTensor([x])          #pytorch形式输入
seqence_output, pooler_output = bert(torch_x)
print(seqence_output.shape, pooler_output.shape)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))

class ModelBert:
    def __init__(self, state_dict):
        self.num_attention_size = 12
        self.num_layers = 1
        self.hidden_size = 768
        self.wight = self.load_weight(state_dict)


    def load_weight(self,state_dict):
        # embedding部分 3层
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy()  # 基础embedding层，用于将输入的句子转换为词向量
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()  # 位置embedding层，记录句子中每个词的位置
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()  # 词性embedding层，记录句子中每个词的类型
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()  # embedding层归一化的收缩权重
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()  # embedding层归一化的偏移量
        self.all_weight = []  # 用于记录state_dict中训练好的所有参数
        for i in range(self.num_layers):
            # 自注意机制 3层 attention(q,k,v)
            q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % i].numpy()  # 线性层query收缩权重
            q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % i].numpy()   # 线性层query偏移量
            k_w = state_dict["encoder.layer.%d.attention.self.key.weight" % i].numpy()  # 线性层key收缩权重
            k_b = state_dict["encoder.layer.%d.attention.self.key.bias" % i].numpy()   # 线性层key偏移量
            v_w = state_dict["encoder.layer.%d.attention.self.value.weight" % i].numpy()  # 线性层value收缩权重
            v_b = state_dict["encoder.layer.%d.attention.self.value.bias" % i].numpy()   # 线性层value偏移量
            # 自注意机制后经过一个线性层 [max_len, hidden_size]
            attention_output_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % i].numpy()  # 自注意机制的输出层权重
            attention_output_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % i].numpy()    # 自注意机制的输出层偏移量
            # 经过线性层后进行归一化->残差机制 [max_len, hidden_size]
            attention_layer_norm_weight = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i].numpy()  # 自注意机制的归一化层权重
            attention_layer_norm_bias = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i].numpy()      # 自注意机制的归一化层偏移量
            # feed forward层 [max_len, hidden_size]
            intermediate_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % i].numpy()  # feed forward输入层权重
            intermediate_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % i].numpy()    # feed forward输入层偏移量
            output_weight = state_dict["encoder.layer.%d.output.dense.weight" % i].numpy()  # feed forward输出层权重
            output_bias = state_dict["encoder.layer.%d.output.dense.bias" % i].numpy()    # feed forward输出层偏移量
            # 再次归一化处理
            output_layer_norm_weight = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i].numpy() # feed forward的归一化层权重
            output_layer_norm_bias = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i].numpy()  # feed forward的归一化层偏移量
            self.all_weight.append([q_w, q_b, k_w, k_b, v_w, v_b,
                                    attention_output_weight, attention_output_bias,
                                    attention_layer_norm_weight, attention_layer_norm_bias,
                                    intermediate_weight, intermediate_bias, output_weight, output_bias,
                                    output_layer_norm_weight, output_layer_norm_bias])
            # pool层
            self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()  # 池化层代替了原来Segment embedding的作用
            self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()

    def embeddings_forward(self, x):
        word_embedding = self.GetEmbedding(self.word_embeddings, x)  # [batch_size, max_len, hidden_size]
        position_embedding = self.GetEmbedding(self.position_embeddings, np.array(list(range(len(x)))))
        token_type_embedding = self.GetEmbedding(self.token_type_embeddings, np.array([0] * len(x)))
        #加和
        embedding = word_embedding + position_embedding + token_type_embedding
        #归一化
        embeddings = self.layer_norm(embedding, self.embeddings_layer_norm_weight, self.embeddings_layer_norm_bias)
        return embeddings

    def all_transformer_layer_forward(self, x):
        for i in range(self.num_layers):
            x = self.single_transformer_layer_forward(x, i)
        return x

    def single_transformer_layer_forward(self, x, layer_index):
        wights = self.all_weight[layer_index]
        q_w, q_b, \
        k_w, k_b, \
        v_w, v_b, \
        attention_output_weight, attention_output_bias, \
        attention_layer_norm_weight, attention_layer_norm_bias, \
        intermediate_weight, intermediate_bias,\
        output_weight, output_bias,\
        output_layer_norm_weight, output_layer_norm_bias = wights
        attention_out = self.attention_forward(x, q_w, q_b, k_w, k_b, v_w, v_b,
                                            attention_output_weight, attention_output_bias,
                                            self.num_attention_size, self.hidden_size)
        x = self.layer_norm(x+attention_out, attention_layer_norm_weight, attention_layer_norm_bias)
        feed_forward_out = self.feed_forward_forward(x, intermediate_weight, intermediate_bias,
                                                     output_weight, output_bias)
        x = self.layer_norm(x+feed_forward_out, output_layer_norm_weight, output_layer_norm_bias)
        return x

    def GetEmbedding(self, embedding_matrix, x):
        return np.array([embedding_matrix[index] for index in x])

    def layer_norm(self, x, w, b):
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        x = x * w + b
        return x

    def attention_forward(self, x, q_w, q_b, k_w, k_b, v_w, v_b,
                          attention_output_weight, attention_output_bias,
                          num_attention_size, hidden_size):
        q = np.dot(x, q_w.T) + q_b
        k = np.dot(x, k_w.T) + k_b
        v = np.dot(x, v_w.T) + v_b
        attention_head_size = int(hidden_size / num_attention_size)
        q = self.transpose_for_scores(q, num_attention_size, attention_head_size)  # [batch_size, num_attention_size, max_len, attention_head_size]
        k = self.transpose_for_scores(k, num_attention_size, attention_head_size)
        v = self.transpose_for_scores(v, num_attention_size, attention_head_size)
        qk = np.matmul(q, k.swapaxes(1, 2))   # [batch_size, num_attention_size, max_len, max_len]
        qk /= np.sqrt(attention_head_size) # 缩放
        qk = softmax(qk) # [batch_size, num_attention_size, max_len, max_len]
        qkv = np.matmul(qk, v)    # [batch_size, num_attention_size, max_len, attention_head_size]
        qkv = qkv.swapaxes(0, 1).reshape(-1, hidden_size)
        attention = np.dot(qkv, attention_output_weight.T) + attention_output_bias
        return attention

    # 多头机制
    def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
        max_len, hidden_size = x.shape
        x = x.reshape(max_len, num_attention_heads, attention_head_size)
        x = x.swapaxes(0, 1) # [num_attention_heads, max_len, attention_head_size]
        return x

    def feed_forward_forward(self, x, intermediate_weight, intermediate_bias, output_weight, output_bias):
        x = np.dot(x, intermediate_weight.T) + intermediate_bias
        x = gelu(x)
        x = np.dot(x, output_weight.T) + output_bias
        return x

    def pool_forward(self, x):
        x = np.dot(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
        x = np.tanh(x)
        return x

    def forward(self, x):
         x = self.embeddings_forward(x)
         sequence_output = self.all_transformer_layer_forward(x)
         pool_output = self.pool_forward(sequence_output[0])
         return sequence_output, pool_output

    def num_trainable_parameters(self, x):
        x = self.GetEmbedding(self.word_embeddings, x)
        max_len, hidden_size = x.shape
        em = (max_len * hidden_size) * 2 + 512 * hidden_size
        att = (max_len * hidden_size) * 3
        att_out_liner = (max_len * hidden_size)
        feed = (max_len * hidden_size) * 2
        pool = (max_len * hidden_size)
        total = sum([em, att, att_out_liner, feed, pool])
        return total


db = ModelBert(state_dicts)
diy_sequence_output, diy_pooler_output = db.forward(x)
#torch
torch_sequence_output, torch_pooler_output = bert(torch_x)
num = db.num_trainable_parameters(x)
print("可训练参数量：%d" % num)
print(diy_sequence_output)
print(torch_sequence_output)
