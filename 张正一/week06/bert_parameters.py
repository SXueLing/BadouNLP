import torch
import math
import numpy as np
from transformers import BertModel

bert = BertModel.from_pretrained('bert-base-uncased', return_dict=False)
state_dict = bert.state_dict()
# print(state_dict)
bert.eval()
x = np.array([2450, 15486, 102, 2110])
torched_x = torch.LongTensor([x])
# print(torched_x)
sequence_output, pooled_output = bert(torched_x)
print(14)
print(sequence_output)
# print(bert.state_dict().keys())

#softmax归一化
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)

# 激活函数
def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))

class MyBert:
    
    def __init__(self, state_dict):
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = 1
        self.load_weights(state_dict)
    
    def load_weights(self,state_dict):
        # embeddings部分
        # 词汇表大小
        self.word_embeddings = state_dict['embeddings.word_embeddings.weight'].numpy()  # (30522, 768) 
        # 位置信息
        self.position_embeddings = state_dict['embeddings.position_embeddings.weight'].numpy()  # (512, 768)
        # 句子类型
        self.token_type_embeddings = state_dict['embeddings.token_type_embeddings.weight'].numpy()  # (2, 768)
        # 归一化后乘以权重
        self.LayerNorm_weight = state_dict['embeddings.LayerNorm.weight'].numpy()  # (768,)
        # 归一化后加偏置
        self.LayerNorm_bias = state_dict['embeddings.LayerNorm.bias'].numpy()  # (768,)
        
        self.transformer_weights = []
        # 多层Transformer
        for i in range(self.num_layers):
            q_w = state_dict[f'encoder.layer.{i}.attention.self.query.weight'].numpy()  # (768, 768)
            q_b = state_dict[f'encoder.layer.{i}.attention.self.query.bias'].numpy()  # (768,)
            k_w = state_dict[f'encoder.layer.{i}.attention.self.key.weight'].numpy()  # (768, 768)
            k_b = state_dict[f'encoder.layer.{i}.attention.self.key.bias'].numpy()  # (768,)
            v_w = state_dict[f'encoder.layer.{i}.attention.self.value.weight'].numpy()  # (768, 768)
            v_b = state_dict[f'encoder.layer.{i}.attention.self.value.bias'].numpy()  # (768,)
            attention_output_w = state_dict[f'encoder.layer.{i}.attention.output.dense.weight'].numpy()  # (768, 768)
            attention_output_b = state_dict[f'encoder.layer.{i}.attention.output.dense.bias'].numpy()  # (768,)
            attention_output_LayerNorm_weight = state_dict[f'encoder.layer.{i}.attention.output.LayerNorm.weight'].numpy()  # (768,)
            attention_output_LayerNorm_bias = state_dict[f'encoder.layer.{i}.attention.output.LayerNorm.bias'].numpy()  # (768,)
            intermediate_w = state_dict[f'encoder.layer.{i}.intermediate.dense.weight'].numpy()  # (3072, 768)
            intermediate_b = state_dict[f'encoder.layer.{i}.intermediate.dense.bias'].numpy()  # (3072,)
            output_w = state_dict[f'encoder.layer.{i}.output.dense.weight'].numpy()  # (768, 3072)
            output_b = state_dict[f'encoder.layer.{i}.output.dense.bias'].numpy()  # (768,)
            output_LayerNorm_weight = state_dict[f'encoder.layer.{i}.output.LayerNorm.weight'].numpy()  # (768,)
            output_LayerNorm_bias = state_dict[f'encoder.layer.{i}.output.LayerNorm.bias'].numpy()  # (768,)
            self.transformer_weights.append((q_w, q_b, k_w, k_b, v_w, v_b, attention_output_w, attention_output_b, attention_output_LayerNorm_weight, attention_output_LayerNorm_bias, intermediate_w, intermediate_b, output_w, output_b, output_LayerNorm_weight, output_LayerNorm_bias))
        # 输出层
        self.dense_weight = state_dict['pooler.dense.weight'].numpy()  # (768, 768)
        self.dense_bias = state_dict['pooler.dense.bias'].numpy()  # (768,)
    def get_embedding(self, embedding_matrix, x):
        return np.array([embedding_matrix[i] for i in x])
    
    # 归一化
    def layer_norm(self, x, w, b):
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        x = x * w + b #这里的计算过程是将w和b先进行broadcast，然后进行矩阵逐元素乘法和加法运算
        return x
    
    # 前向传播 
    def embedding_forward(self, x):
        we = self.get_embedding(self.word_embeddings, x)  # (batch_size, seq_len, 768)
        pe = self.get_embedding(self.position_embeddings, np.array(list(range(len(x)))))  # (batch_size, seq_len, 768)
        te = self.get_embedding(self.token_type_embeddings, np.array([0]*len(x)))  # (batch_size, seq_len, 768)
        embedding = we + pe + te  # (batch_size, seq_len, 768)
        embedding = self.layer_norm(embedding, self.LayerNorm_weight, self.LayerNorm_bias)  # (batch_size, seq_len, 768)
        return embedding
    # 多头机制
    def transpose_for_scores(self, x, attention_head_size, num_attention_heads):
        sentence_len, hidden_size = x.shape
        x = x.reshape(sentence_len, num_attention_heads, attention_head_size)  # (sentence_len, num_attention_heads, attention_head_size)
        x = x.swapaxes(0, 1)  # (num_attention_heads, sentence_len, attention_head_size)
        return x
    
    # 自注意力层  
    def self_attention(self, x, q_w, q_b, k_w, k_b, v_w, v_b, attention_output_w, attention_output_b, num_attention_heads, hidden_size):
        q = np.dot(x, q_w.T) + q_b # (sentence_len, hidden_size)
        k = np.dot(x, k_w.T) + k_b # (sentence_len, hidden_size)
        v = np.dot(x, v_w.T) + v_b # (sentence_len, hidden_size)
        attention_head_size = int(hidden_size / num_attention_heads)
        q = self.transpose_for_scores(q, attention_head_size, num_attention_heads) # (num_attention_heads, sentence_len, attention_head_size)
        k = self.transpose_for_scores(k, attention_head_size, num_attention_heads) # (num_attention_heads, sentence_len, attention_head_size)
        v = self.transpose_for_scores(v, attention_head_size, num_attention_heads) # (num_attention_heads, sentence_len, attention_head_size)
        qk = np.matmul(q, k.swapaxes(1, 2))  # (num_attention_heads, sentence_len, sentence_len)
        qk = qk / math.sqrt(attention_head_size)  # (num_attention_heads, sentence_len, sentence_len)
        qk = softmax(qk)  # (num_attention_heads, sentence_len, sentence_len)
        qkv = np.matmul(qk, v)  # (num_attention_heads, sentence_len, attention_head_size)
        qkv = qkv.swapaxes(0, 1).reshape(-1, hidden_size)  # (sentence_len, hidden_size)
        attention = np.dot(qkv, attention_output_w.T) + attention_output_b  # (sentence_len, hidden_size)
        return attention
        
    # 前馈网络
    def feed_forward(self, x, intermediate_w, intermediate_b, output_w, output_b):
        x = np.dot(x, intermediate_w.T) + intermediate_b  # (sentence_len, 3072)
        x = gelu(x)  # (sentence_len, 3072)
        x = np.dot(x, output_w.T) + output_b  # (sentence_len, hidden_size)
        return x
        
    # 执行单个transformer层的前向传播
    def single_transformer_layer_forward(self, x, layer_index):
        weights = self.transformer_weights[layer_index] 
        q_w, q_b, k_w, k_b, v_w, v_b, attention_output_w, attention_output_b, attention_output_LayerNorm_weight, attention_output_LayerNorm_bias, intermediate_w, intermediate_b, output_w, output_b, output_LayerNorm_weight, output_LayerNorm_bias = weights

        attention_output = self.self_attention(x, q_w, q_b, k_w, k_b, v_w, v_b, attention_output_w, attention_output_b, self.num_attention_heads, self.hidden_size)  # (sentence_len, hidden_size)

        # bn层 残差机制 和归一化
        x = self.layer_norm(x + attention_output, attention_output_LayerNorm_weight, attention_output_LayerNorm_bias)  # (sentence_len, hidden_size)

        # 前馈网络
        feed_forward_x = self.feed_forward(x, intermediate_w, intermediate_b, output_w, output_b)  # (sentence_len, hidden_size)
        
        # bn层 残差机制 和归一化
        x = self.layer_norm(x + feed_forward_x, output_LayerNorm_weight, output_LayerNorm_bias)  # (sentence_len, hidden_size)
        
        return x
        
        
    # 全部tranformer层的前向传播
    def all_transformer_layer_forward(self, x):
        for i in range(self.num_layers):
            x = self.single_transformer_layer_forward(x, i)
        return x
            
    #链接[cls] token的输出层
    def pooler_output_layer(self, x):
        x = np.dot(x, self.dense_weight.T) + self.dense_bias
        x = np.tanh(x)
        return x

    def forward(self, x):
        x = self.embedding_forward(x)  # (seq_len, 768)
        sequence_output = self.all_transformer_layer_forward(x)  # (seq_len, hidden_size)
        pooled_output = self.pooler_output_layer(sequence_output[0])
        return sequence_output, pooled_output
    
    def get_model_params(self, n = 4, vocab = 30522, max_sequence_length = 512, embedding_size = 768, hide_size = 3072, num_layers = 1):
        # embedding过程中的参数，其中 vocab * embedding_size是词表embedding参数， max_sequence_length * embedding_size是位置参数， n * embedding_size是句子参数
        # embedding_size + embedding_sizes是layer_norm层参数
        embedding_parameters = vocab * embedding_size + max_sequence_length * embedding_size + n * embedding_size + embedding_size + embedding_size

        # self_attention过程的参数, 其中embedding_size * embedding_size是权重参数，embedding_size是bias， *3是K Q V三个
        self_attention_parameters = (embedding_size * embedding_size + embedding_size) * 3

        # self_attention_out参数 其中 embedding_size * embedding_size + embedding_size + embedding_size是self输出的线性层参数，embedding_size + embedding_size是layer_norm层参数
        self_attention_out_parameters = embedding_size * embedding_size + embedding_size + embedding_size + embedding_size

        # Feed Forward参数 其中embedding_size * hide_size + hide_size第一个线性层，embedding_size * hide_size + embedding_size第二个线性层，
        # embedding_size + embedding_size是layer_norm层
        feed_forward_parameters = embedding_size * hide_size + hide_size + embedding_size * hide_size + embedding_size + embedding_size + embedding_size

        # pool_fc层参数
        pool_fc_parameters = embedding_size * embedding_size + embedding_size

        # 模型总参数 = embedding层参数 + self_attention参数 + self_attention_out参数 + Feed_Forward参数 + pool_fc层参数
        all_paramerters = embedding_parameters + (self_attention_parameters + self_attention_out_parameters + feed_forward_parameters) * num_layers + pool_fc_parameters
        print(all_paramerters)
        
    

my_bert = MyBert(state_dict)
my_sequence_output, my_pooled_output = my_bert.forward(x)
my_bert.get_model_params()
print(159)
print(my_sequence_output)
