import torch
import math
import numpy as np
from transformers import BertModel

'''

通过手动矩阵运算实现Bert结构
模型文件下载 https://huggingface.co/models

'''

bert = BertModel.from_pretrained(r"D:\AI\homeWork\week06\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
bert.eval()
x = np.array([2450, 15486, 102, 2110])   #假想成4个字的句子
torch_x = torch.LongTensor([x])          #pytorch形式输入
seqence_output, pooler_output = bert(torch_x)
print(seqence_output.shape, pooler_output.shape)
# print(seqence_output, pooler_output)

print(bert.state_dict().keys())  #查看所有的权值矩阵名称

#softmax归一化
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)

#gelu激活函数
def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))

class DiyBert:
    #将预训练好的整个权重字典输入进来
    def __init__(self, state_dict):
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = 1        #注意这里的层数要跟预训练config.json文件中的模型层数一致
        self.load_weights(state_dict)

    def load_weights(self, state_dict):
        #embedding部分
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy()
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()
        self.transformer_weights = []

        #transformer部分，有多层
        for i in range(self.num_layers):
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

            self.transformer_weights.append([q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                                             attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias,
                                             output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])
        #pooler层
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()
        embedding_para = self.word_embeddings.size +  self.position_embeddings.size + self.token_type_embeddings.size + self.embeddings_layer_norm_weight.size +\
                      self.embeddings_layer_norm_bias.size
        q_w, q_b, \
        k_w, k_b, \
        v_w, v_b, \
        attention_output_weight, attention_output_bias, \
        attention_layer_norm_w, attention_layer_norm_b, \
        intermediate_weight, intermediate_bias, \
        output_weight, output_bias, \
        ff_layer_norm_w, ff_layer_norm_b = self.transformer_weights[0]

        transformer_para = q_w.size + q_b.size + k_w.size + k_b.size + v_w.size + v_b.size + \
                           attention_output_weight.size + attention_output_bias.size + \
                           attention_layer_norm_w.size + attention_layer_norm_b.size + \
                           intermediate_weight.size + intermediate_bias.size + \
                           output_weight.size + output_bias.size + \
                           ff_layer_norm_w.size + ff_layer_norm_b.size

        print(f"total_para:{embedding_para + transformer_para} = embedding_para:{embedding_para} + transformer_para:{transformer_para}")
        total_parameter_1 = (21128 + 512 + 2) * self.hidden_size + self.hidden_size * 2 +\
                            self.num_layers * (3 * (self.hidden_size + 1) * self.hidden_size + (self.hidden_size + 1) * self.hidden_size +\
                                               2 * self.hidden_size +\
                                               2 * (self.hidden_size) * (self.hidden_size * 4) + \
                                               (self.hidden_size) + (self.hidden_size * 4) + \
                                               2 * self.hidden_size)
        print(f"total_parameter_1:{total_parameter_1}, diff:{embedding_para + transformer_para - total_parameter_1}")

    #bert embedding，使用3层叠加，在经过一个Layer norm层
    def embedding_forward(self, x):
        # x.shape = [max_len]
        we = self.get_embedding(self.word_embeddings, x)  # shpae: [max_len, hidden_size]
        print(f"word_embeddings.shape:{self.word_embeddings.shape}, word_embeddings.size:{self.word_embeddings.size}")
        # position embeding的输入 [0, 1, 2, 3]
        pe = self.get_embedding(self.position_embeddings, np.array(list(range(len(x)))))  # shpae: [max_len, hidden_size]
        print(f"position_embeddings.shape:{self.position_embeddings.shape}, position_embeddings.size:{self.position_embeddings.size}")
        # token type embedding,单输入的情况下为[0, 0, 0, 0]
        te = self.get_embedding(self.token_type_embeddings, np.array([0] * len(x)))  # shpae: [max_len, hidden_size]
        print(f"token_type_embeddings.shape:{self.token_type_embeddings.shape}, token_type_embeddings.size:{self.token_type_embeddings.size}")
        embedding = we + pe + te
        # 加和后有一个归一化层
        embedding = self.layer_norm(embedding, self.embeddings_layer_norm_weight, self.embeddings_layer_norm_bias)  # shpae: [max_len, hidden_size]
        print(f"embeddings_layer_norm_weight.shape:{self.embeddings_layer_norm_weight.shape}, embeddings_layer_norm_weight.size:{self.embeddings_layer_norm_weight.size}")
        print(f"embeddings_layer_norm_bias.shape:{self.embeddings_layer_norm_bias.shape}, embeddings_layer_norm_bias.size:{self.embeddings_layer_norm_bias.size}")
        return embedding

    #embedding层实际上相当于按index索引，或理解为onehot输入乘以embedding矩阵
    def get_embedding(self, embedding_matrix, x):
        return np.array([embedding_matrix[index] for index in x])

    #执行全部的transformer层计算
    def all_transformer_layer_forward(self, x):
        for i in range(self.num_layers):
            x = self.single_transformer_layer_forward(x, i)
        return x

    #执行单层transformer层计算
    def single_transformer_layer_forward(self, x, layer_index):
        weights = self.transformer_weights[layer_index]
        #取出该层的参数，在实际中，这些参数都是随机初始化，之后进行预训练
        q_w, q_b, \
        k_w, k_b, \
        v_w, v_b, \
        attention_output_weight, attention_output_bias, \
        attention_layer_norm_w, attention_layer_norm_b, \
        intermediate_weight, intermediate_bias, \
        output_weight, output_bias, \
        ff_layer_norm_w, ff_layer_norm_b = weights
        #self attention层
        attention_output = self.self_attention(x,
                                q_w, q_b,
                                k_w, k_b,
                                v_w, v_b,
                                attention_output_weight, attention_output_bias,
                                self.num_attention_heads,
                                self.hidden_size)
        #bn层，并使用了残差机制
        x = self.layer_norm(x + attention_output, attention_layer_norm_w, attention_layer_norm_b)
        print(f"attention_layer_norm_w.shape:{attention_layer_norm_w.shape}, attention_layer_norm_w.size:{attention_layer_norm_w.size}")
        print(f"attention_layer_norm_b.shape:{attention_layer_norm_b.shape}, attention_layer_norm_b.size:{attention_layer_norm_b.size}")
        #feed forward层
        feed_forward_x = self.feed_forward(x,
                              intermediate_weight, intermediate_bias,
                              output_weight, output_bias)
        #bn层，并使用了残差机制
        x = self.layer_norm(x + feed_forward_x, ff_layer_norm_w, ff_layer_norm_b)
        print(f"ff_layer_norm_w.shape:{ff_layer_norm_w.shape}, ff_layer_norm_w.size:{ff_layer_norm_w.size}")
        print(f"ff_layer_norm_b.shape:{ff_layer_norm_b.shape}, ff_layer_norm_b.size:{ff_layer_norm_b.size}")
        return x

    # self attention的计算
    def self_attention(self,
                       x,
                       q_w,
                       q_b,
                       k_w,
                       k_b,
                       v_w,
                       v_b,
                       attention_output_weight,
                       attention_output_bias,
                       num_attention_heads,
                       hidden_size):
        # x.shape = max_len * hidden_size
        # q_w, k_w, v_w  shape = hidden_size * hidden_size
        # q_b, k_b, v_b  shape = hidden_size
        q = np.dot(x, q_w.T) + q_b  # shape: [max_len, hidden_size]      W * X + B lINER
        print(f"q_w.shape:{q_w.shape}, q_w.size:{q_w.size}")
        print(f"q_b.shape:{q_b.shape}, q_b.size:{q_b.size}")
        k = np.dot(x, k_w.T) + k_b  # shpae: [max_len, hidden_size]
        print(f"k_w.shape:{k_w.shape}, k_w.size:{k_w.size}")
        print(f"k_b.shape:{k_b.shape}, k_b.size:{k_b.size}")
        v = np.dot(x, v_w.T) + v_b  # shpae: [max_len, hidden_size]
        print(f"v_w.shape:{v_w.shape}, v_w.size:{v_w.size}")
        print(f"v_b.shape:{v_b.shape}, v_b.size:{v_b.size}")
        attention_head_size = int(hidden_size / num_attention_heads)
        # q.shape = num_attention_heads, max_len, attention_head_size
        q = self.transpose_for_scores(q, attention_head_size, num_attention_heads)
        # k.shape = num_attention_heads, max_len, attention_head_size
        k = self.transpose_for_scores(k, attention_head_size, num_attention_heads)
        # v.shape = num_attention_heads, max_len, attention_head_size
        v = self.transpose_for_scores(v, attention_head_size, num_attention_heads)
        # qk.shape = num_attention_heads, max_len, max_len
        qk = np.matmul(q, k.swapaxes(1, 2))  
        qk /= np.sqrt(attention_head_size)
        qk = softmax(qk)
        # qkv.shape = num_attention_heads, max_len, attention_head_size
        qkv = np.matmul(qk, v)
        # qkv.shape = max_len, hidden_size
        qkv = qkv.swapaxes(0, 1).reshape(-1, hidden_size)
        # attention.shape = max_len, hidden_size
        attention = np.dot(qkv, attention_output_weight.T) + attention_output_bias
        print(f"attention_output_weight.shape:{attention_output_weight.shape}, attention_output_weight.size:{attention_output_weight.size}")
        print(f"attention_output_bias.shape:{attention_output_bias.shape}, attention_output_bias.size:{attention_output_bias.size}")
        return attention

    #多头机制
    def transpose_for_scores(self, x, attention_head_size, num_attention_heads):
        # hidden_size = 768  num_attent_heads = 12 attention_head_size = 64
        max_len, hidden_size = x.shape
        x = x.reshape(max_len, num_attention_heads, attention_head_size)
        x = x.swapaxes(1, 0)  # output shape = [num_attention_heads, max_len, attention_head_size]
        return x

    #前馈网络的计算
    def feed_forward(self,
                     x,
                     intermediate_weight,  # intermediate_size, hidden_size
                     intermediate_bias,  # intermediate_size
                     output_weight,  # hidden_size, intermediate_size
                     output_bias,  # hidden_size
                     ):
        # output shpae: [max_len, intermediate_size]
        x = np.dot(x, intermediate_weight.T) + intermediate_bias
        print(f"intermediate_weight.shape:{intermediate_weight.shape}, intermediate_weight.size:{intermediate_weight.size}")
        print(f"intermediate_bias.shape:{intermediate_bias.shape}, intermediate_bias.size:{intermediate_bias.size}")
        x = gelu(x)
        # output shpae: [max_len, hidden_size]
        x = np.dot(x, output_weight.T) + output_bias
        print(f"output_weight.shape:{output_weight.shape}, output_weight.size:{output_weight.size}")
        print(f"output_bias.shape:{output_bias.shape}, output_bias.size:{output_bias.size}")
        return x

    #归一化层
    def layer_norm(self, x, w, b):
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        x = x * w + b
        return x

    #链接[cls] token的输出层
    def pooler_output_layer(self, x):
        x = np.dot(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
        print(f"pooler_dense_weight.shape:{self.pooler_dense_weight.shape}, pooler_dense_weight.size:{self.pooler_dense_weight.size}")
        print(f"pooler_dense_bias.shape:{self.pooler_dense_bias.shape}, pooler_dense_bias.size:{self.pooler_dense_bias.size}")
        x = np.tanh(x)
        return x

    #最终输出
    def forward(self, x):
        x = self.embedding_forward(x)
        sequence_output = self.all_transformer_layer_forward(x)
        pooler_output = self.pooler_output_layer(sequence_output[0])
        return sequence_output, pooler_output


#自制
db = DiyBert(state_dict)
diy_sequence_output, diy_pooler_output = db.forward(x)
#torch
torch_sequence_output, torch_pooler_output = bert(torch_x)

print(diy_sequence_output)
print(torch_sequence_output)

# print(diy_pooler_output)
# print(torch_pooler_output)
