#bert 模型参数

import torch 
import math 
import numpy as np 
from transformers import BertModel

bert = BertModel.from_pretrained("bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
bert.eval()
x = np.array([2450, 15486, 102, 2110])
torch_x = torch.LongTensor([x])
sequence_output, pooler_output = bert(torch_x)
print(sequence_output.shape, pooler_output.shape)
print(bert.state_dict().keys())

"""
odict_keys(['embeddings.word_embeddings.weight'
, 'embeddings.position_embeddings.weight'
, 'embeddings.token_type_embeddings.weight'
, 'embeddings.LayerNorm.weight'
, 'embeddings.LayerNorm.bias'
, 'encoder.layer.0.attention.self.query.weight'
, 'encoder.layer.0.attention.self.query.bias'
, 'encoder.layer.0.attention.self.key.weight'
, 'encoder.layer.0.attention.self.key.bias'
, 'encoder.layer.0.attention.self.value.weight'
, 'encoder.layer.0.attention.self.value.bias'
, 'encoder.layer.0.attention.output.dense.weight'
, 'encoder.layer.0.attention.output.dense.bias'
, 'encoder.layer.0.attention.output.LayerNorm.weight'
, 'encoder.layer.0.attention.output.LayerNorm.bias'
, 'encoder.layer.0.intermediate.dense.weight'
, 'encoder.layer.0.intermediate.dense.bias'
, 'encoder.layer.0.output.dense.weight'
, 'encoder.layer.0.output.dense.bias'
,'encoder.layer.0.output.LayerNorm.weight'
 , 'encoder.layer.0.output.LayerNorm.bias'
 , 'pooler.dense.weight'
 ,'pooler.dense.bias'])
"""

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))

class DiyBert:
    def __init__(self, state_dict):
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = 12
        self.load_weights(state_dict)
    
    def load_weights(self, state_dict):
        #embedding
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy()
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
        self.embedding_layer_norm_bais = state_dict["embeddings.LayerNorm.bias"].numpy()
        #transform 
        self.transformer_wights = []
        for i in range(self.num_layers):
            q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % i].numpy()
            q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % i].numpy()
            k_w = state_dict["encoder.layer.%d.attention.self.key.weight" % i].numpy()
            k_b = state_dict["encoder.layer.%d.attention.self.key.bias" % i].numpy()
            v_w = state_dict["encoder.layer.%d.attention.self.value.weight" % i].numpy()
            v_b = state_dict["encoder.layer.%d.attention.self.value.bias" % i].numpy()
            attention_output_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % i].numpy()
            attention_output_bais = state_dict["encoder.layer.%d.attention.output.dense.bias" % i].numpy()
            attention_layer_norm_w = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i].numpy()
            attention_layer_norm_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i].numpy()
            intermediate_w = state_dict["encoder.layer.%d.intermediate.dense.weight" % i].numpy()
            intermediate_b = state_dict["encoder.layer.%d.intermediate.dense.bias" % i].numpy()
            output_w = state_dict["encoder.layer.%d.output.dense.weight" % i].numpy()
            output_b = state_dict["encoder.layer.%d.output.dense.bias" % i].numpy()
            ff_layer_norm_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i].numpy()
            ff_layer_norm_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i].numpy()
            self.transformer_wights.append([q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bais
                                            , attention_layer_norm_w, attention_layer_norm_b, intermediate_w, intermediate_b
                                            , output_w, output_b
                                            , ff_layer_norm_w, ff_layer_norm_b])
            #pooler
            self.pooler_dense_w = state_dict["pooler.dense.weight"].numpy()
            self.pooler_dense_b = state_dict["pooler.dense.bias"].numpy()

    def embedding_forward(self, x):
        we = self.get_embedding(self.word_embeddings, x) #[len, hidden_size]
        pe = self.get_embedding(self.position_embeddings, np.array(list(range(len(x))))) #[len, hidden_size]
        te = self.get_embedding(self.token_type_embeddings, np.array([0]*len(x))) #[len, hidden_size]
        embedding = we + pe + te 
        embedding = self.layer_norm(embedding, self.embeddings_layer_norm_weight, self.embedding_layer_norm_bais) #[en, hidden_size]
        return embedding
    
    def get_embedding(self, embedding_matrix, x):
        return np.array([embedding_matrix[index] for index in x])
    
    def all_transformer_layer_forward(self, x):
        for i in range(self.num_layers):
            x = self.single_transformer_layer_forward(x, i)
        return x 
    
    def single_transformer_layer_forward(self, x, layer_index):
        weights = self.transformer_wights[layer_index]
        q_w, q_b, \
        k_w, k_b, \
        v_w, v_b, \
        attention_output_weight, attention_output_bais, \
        attention_layer_norm_w, attention_layer_norm_b, \
        intermediate_w, intermediate_b, \
        output_weight, output_bias, \
        ff_layer_norm_w, ff_layer_norm_b = weights
        #self attention
        attention_output = self.self_attention(x,
                                               q_w, q_b,
                                               k_w, k_b,
                                               v_w, v_b,
                                               attention_output_weight, attention_output_bais,
                                               self.num_attention_heads, self.hidden_size)
        #bn
        x = self.layer_norm(x + attention_output, attention_layer_norm_w, attention_layer_norm_b)
        #feed forward
        feed_forward_x = self.feed_forward(x,
                                           intermediate_w, intermediate_b,
                                           output_weight, output_bias)
        #bn 
        x = self.layer_norm(x + feed_forward_x, ff_layer_norm_w, ff_layer_norm_b)
        return x 
        
    def self_attention(self, x, q_w, q_b, k_w, k_b, v_w, v_b,
                       attention_output_weight, attention_output_bais,
                       num_attention_heads, hidden_size):
        q = np.dot(x, q_w.T) + q_b #[len, hidden_size]
        k = np.dot(x, k_w.T) + k_b #[len, hidden_size]
        v = np.dot(x, v_w.T) + v_b #[len, hidden_size]
        attention_head_size = int(hidden_size / num_attention_heads)
        q = self.transpose_for_scores(q, attention_head_size, num_attention_heads)
        k = self.transpose_for_scores(k, attention_head_size, num_attention_heads)
        v = self.transpose_for_scores(v, attention_head_size, num_attention_heads)
        qk = np.matmul(q, k.swapaxes(1, 2))
        qk /= np.sqrt(attention_head_size)
        qk = softmax(qk)
        qkv = np.matmul(qk, v)
        qkv = qkv.swapaxes(0, 1).reshape(-1, hidden_size)
        attention = np.dot(qkv, attention_output_weight.T) + attention_output_bais
        return attention
    
    def feed_forward(self, x, 
                     intermediate_w, intermediate_b,
                     output_weight, output_bias):
        x = np.dot(x, intermediate_w.T) + intermediate_b
        x = gelu(x)
        x = np.dot(x, output_weight.T) + output_bias
        return x
    
    def transpose_for_scores(self, x, attention_head_size, num_attention_heads):
        max_len, hidden_size = x.shape
        x = x.reshape(max_len, num_attention_heads, attention_head_size)
        x = x.swapaxes(1, 0) #[num_attention_heads, max_len, attention_head_size]
        return x
    
    def layer_norm(self, x, w, b):
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        x = x * w + b 
        return x
    
    #[cls]
    def pooler_output_layer(self, x):
        x = np.dot(x, self.pooler_dense_w.T) + self.pooler_dense_b
        x = np.tanh(x)
        return x 
    
    def forward(self, x):
        x = self.embedding_forward(x)
        sequence_output = self.all_transformer_layer_forward(x)
        pooler_output = self.pooler_output_layer(sequence_output[0])
        return sequence_output, pooler_output
    
    def countpara(self, state_dict):
        para_cnt = 0
        for key in state_dict.keys():
            para = state_dict[key].numpy()

            if len(para.shape) == 1:
                para_cnt += len(para)
            else:
                m, n = para.shape
                para_cnt += m*n
        return para_cnt

        

    



db = DiyBert(state_dict)
diy_sequence_output, diy_pooler_output = db.forward(x)
para_cnt = db.countpara(state_dict)
#torch_sequence_output, torch_pooler_output = bert(torch_x)
print(diy_sequence_output)
print(para_cnt)
#print(torch_sequence_output)
