'''
Author: Zhao
Date: 2025-01-02 22:29:24
LastEditTime: 2025-01-03 23:17:45
FilePath: diy_bert.py
Description: 

'''
"""
Author: Zhao
Date: 2025-01-03 14:29:24
LastEditTime: 2025-01-03 15:39:43
FilePath: diy_bert.py
Description: 通过手动矩阵运算实现Bert结构,模型文件下载 https://huggingface.co/models

"""

import torch
import numpy as np
import math
from transformers import BertModel

bert = BertModel.from_pretrained(r"E:\第六周 语言模型\bert-base-chinese", return_dict=False)

#state_dict变量存放训练过程中需要学习的权重和偏执系数
state_dict = bert.state_dict()
bert.eval()
x = np.array([2450, 15486, 102, 2110])  # 假想成4个字的句子
torch_x = torch.LongTensor([x])
seqence_output, pooler_output = bert(torch_x)

"""
    odict_keys(['embeddings.word_embeddings.weight', 'embeddings.position_embeddings.weight', 'embeddings.token_type_embeddings.weight', 
            'embeddings.LayerNorm.weight', 'embeddings.LayerNorm.bias', 
            'encoder.layer.0.attention.self.query.weight', 'encoder.layer.0.attention.self.query.bias', 'encoder.layer.0.attention.self.key.weight', 
            'encoder.layer.0.attention.self.key.bias', 'encoder.layer.0.attention.self.value.weight', 'encoder.layer.0.attention.self.value.bias', 
            'encoder.layer.0.attention.output.dense.weight', 'encoder.layer.0.attention.output.dense.bias', 
            'encoder.layer.0.attention.output.LayerNorm.weight', 'encoder.layer.0.attention.output.LayerNorm.bias', 
            'encoder.layer.0.intermediate.dense.weight', 'encoder.layer.0.intermediate.dense.bias', 
            'encoder.layer.0.output.dense.weight', 'encoder.layer.0.output.dense.bias', 
            'encoder.layer.0.output.LayerNorm.weight', 'encoder.layer.0.output.LayerNorm.bias', 
            'pooler.dense.weight', 'pooler.dense.bias'])
"""
# print(bert.state_dict().keys()) 
print("odict_keys的个数是: ",len(bert.state_dict().keys()))

class DiyBert:
    # 初始化
    def __init__(self,state_dict):
        self.num_attention_heads = 12
        self.num_layers = 1 # 与config.json中num_hidden_layers层一致
        self.hidden_size = 768
        self.load_weights(state_dict)
    
    # 加载所有的待训练参数
    def load_weights(self,state_dict):
        """embedding部分:
            Token Embeddings， (1, n, 768) ，词的向量表示
            Segment Embeddings， (1, n, 768)，辅助BERT区别句子对中的两个句子的向量表示
            Position Embeddings ，(1, n, 768) ，让BERT学习到输入的顺序属性
        """
        
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy()
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()
        self.LayerNorm_weight_embeddings = state_dict["embeddings.LayerNorm.weight"].numpy()
        self.LayerNorm_bias_embeddings = state_dict["embeddings.LayerNorm.bias"].numpy()
        self.transform_weight = [] 
        
        #transformer部分，有多层 对应odict_keys中的encoder 这个与num_hidden_layers层一致
        for i in range(self.num_layers):
            query_weight = state_dict["encoder.layer.%d.attention.self.query.weight" % i].numpy()
            query_bias = state_dict["encoder.layer.%d.attention.self.query.bias" % i].numpy()
            key_weight = state_dict["encoder.layer.%d.attention.self.key.weight" % i].numpy()
            key_bias = state_dict["encoder.layer.%d.attention.self.key.bias" % i].numpy()
            value_weight = state_dict["encoder.layer.%d.attention.self.value.weight" % i].numpy()
            value_bias = state_dict["encoder.layer.%d.attention.self.value.bias" % i].numpy()
            attention_output_dense_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % i].numpy()
            attention_output_dense_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % i].numpy()
            attention_output_LayerNorm_weight = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i].numpy()
            attention_output_LayerNorm_bias = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i].numpy()
            intermediate_dense_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % i].numpy()
            intermediate_dense_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % i].numpy()
            output_dense_weight = state_dict["encoder.layer.%d.output.dense.weight" % i].numpy()
            output_dense_bias = state_dict["encoder.layer.%d.output.dense.bias" % i].numpy()
            output_LayerNorm_weight = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i].numpy()
            output_LayerNorm_bias = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i].numpy()
            self.transform_weight.append([query_weight,query_bias,key_weight,key_bias,value_weight,value_bias,
                                          attention_output_dense_weight,attention_output_dense_bias,
                                          attention_output_LayerNorm_weight,attention_output_LayerNorm_bias,
                                          intermediate_dense_weight,intermediate_dense_bias,
                                          output_dense_weight,output_dense_bias,
                                          output_LayerNorm_weight,output_LayerNorm_bias])
        #pooler层
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()

    """
        word_embeddings,token_type_embeddings,position_embeddingss 三者相加再通过layer normalization
        ---> word_embeddings：即Input后的 TokenEmbedding，为subword对应的嵌入
            token_type_embeddings：即SegmentEmbedding 助区别句子与padding、句子对间的差异
            position_embeddings: 即Position Embeddings 句子中每个词的位置嵌入，用于区别词的顺序,和 transformer论文中的设计不同，
                这里的位置编码是训练出来的，而不是通过Sinusoidal函数计算得到的固定嵌入。这种方法有更好的拓展性可以直接迁移到更长的句子中。
            layer normalization: 可以使得前向传播的输入分布变得稳定，同时使得后向的梯度更加稳定。
    """
    def emmbedding_forward(self, x):
        # x.shape = [max_len] (4,)
        we = self.get_embedding(self.word_embeddings,x=x) # shpae: [max_len, hidden_size](4, 768)
        pe = self.get_embedding(self.position_embeddings,x=np.array(list(range(len(x))))) # shpae: [max_len, hidden_size] (4, 768)
        te = self.get_embedding(self.token_type_embeddings,x=np.array([0] * len(x)))  # shpae: [max_len, hidden_size] (4, 768)
        embedding = we + pe + te # shpae: [max_len, hidden_size] (4, 768)
        # 归一化
        embedding = self.layer_norm(embedding, self.LayerNorm_weight_embeddings,self.LayerNorm_bias_embeddings) # shpae: [max_len, hidden_size] (4, 768)
        return embedding
    
    # 主要计算逻辑层计算
    def all_transform_layer_forward(self, x):
        for i in range(self.num_layers):
            x = self.single_transform_layer_forward(x, i)
        return x
    
    # 取出句子的第一个token（即[CLS]对应的向量），然后过一个全连接层和一个激活函数后输出
    def pooler_out_layer(self, x):
        x = np.dot(x, self.pooler_dense_weight.T) + self.pooler_dense_bias # shpae: [hidden_size] (768,)
        x = np.tanh(x)  # shpae: [hidden_size] (768,)
        return x

    def get_embedding(self, embedding_matrix, x):
        return np.array([embedding_matrix[index] for index in x])
    
    # 定义归一化层计算
    def layer_norm(self, x, w, b):
        # 计算均值
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        # 标准化
        x = x * w + b
        return x
    
    def single_transform_layer_forward(self, x, layer):
        weights = self.transform_weight[layer]
        query_weight, query_bias = weights[0], weights[1]
        key_weight, key_bias = weights[2], weights[3]
        value_weight, value_bias = weights[4], weights[5]
        attention_output_dense_weight, attention_output_dense_bias = weights[6], weights[7]
        attention_output_LayerNorm_weight, attention_output_LayerNorm_bias = weights[8], weights[9]
        intermediate_dense_weight, intermediate_dense_bias = weights[10], weights[11]
        output_dense_weight, output_dense_bias = weights[12], weights[13]
        output_LayerNorm_weight, output_LayerNorm_bias = weights[14], weights[15]

        #计算 self attention
        attention_outputs = self.self_attention(x,query_weight,query_bias,key_weight,key_bias,value_weight,value_bias,
                            attention_output_dense_weight,attention_output_dense_bias,
                            self.num_attention_heads,
                            self.hidden_size
                            )
        # bn层 归一化处理，加和输入数据，解决深层神经网络中的数值不稳定的问题
        x = self.layer_norm(x + attention_outputs, attention_output_LayerNorm_weight, attention_output_LayerNorm_bias)
        #计算 feed forward
        feed_outputs = self.feed_forward(x, intermediate_dense_weight,intermediate_dense_bias,
                          output_dense_weight,output_dense_bias)
        x = self.layer_norm(x + feed_outputs, output_LayerNorm_weight, output_LayerNorm_bias)
        return x
    
    # 计算 self attention
    def self_attention(self, x, 
                       query_weight,
                       query_bias,
                       key_weight,
                       key_bias,
                       value_weight,
                       value_bias,
                       attention_output_dense_weight,
                       attention_output_dense_bias,
                       num_attention_heads,
                       hidden_size
                       ): 
        # x.shape = max_len * hidden_size 4
        # query_weight,key_weight,value_weightd,attention_output_dense_weight 
        # shpae: [ hidden_size * hidden_size] (768, 768) 
        # query_bias,key_bias,value_bias,attention_output_dense_bias
        # shpae: hidden_size (768)
        
        # W * X + B lINER
        query = np.dot(x , query_weight.T) + query_bias # shape: [max_len, hidden_size] (4, 768)
        key = np.dot(x , key_weight.T) + key_bias     # shape: [max_len, hidden_size] (4, 768)
        value = np.dot(x , value_weight.T) + value_bias # shape: [max_len, hidden_size] (4, 768)


        attention_head_size = int(hidden_size / num_attention_heads)
        query_layer = self.multi_head(query,attention_head_size,num_attention_heads)  # shape: [num_attention_heads, max_len, attention_head_size] (12, 4, 64)
        key_layer = self.multi_head(key,attention_head_size,num_attention_heads)  # shape: [num_attention_heads, max_len, attention_head_size] (12, 4, 64)
        value_layer = self.multi_head(value,attention_head_size,num_attention_heads)  # shape: [num_attention_heads, max_len, attention_head_size] (12, 4, 64)


        #将 key 张量的最后一个轴和倒数第二个轴进行转置
        attention_scores = np.matmul(query_layer, key_layer.swapaxes(1, 2)) # shape: [num_attention_heads, max_len, max_len] (12, 4, 4)
        attention_scores = attention_scores / math.sqrt(attention_head_size)
        attention_probs  = softmax(attention_scores)

        context_layer = np.matmul(attention_probs, value_layer) # shape: [num_attention_heads, max_len, attention_head_size] (12, 4, 64)
        context_layer = context_layer.swapaxes(0,1).reshape(-1,hidden_size) # shape: [max_len, hidden_size] (4, 768)
        outputs = np.dot(context_layer, attention_output_dense_weight.T) + attention_output_dense_bias # shape: [max_len, hidden_size] (4, 768)
        return outputs 
    
    # 计算 feed_forward
    def feed_forward(self, x,
                     intermediate_dense_weight, # shape: [intermediate_size, hidden_size] (3072, 768)
                     intermediate_dense_bias,   # shape: [intermediate_size] (3072, )
                     output_dense_weight,       # shape: [hidden_size, intermediate_size] (768, 3072)
                     output_dense_bias          # shape: [intermediate_size] (768, )
                     ):
        x = np.dot(x, intermediate_dense_weight.T) + intermediate_dense_bias # shape: [max_len, intermediate_size] (4, 3072)
        x = gelu(x)
        outputs = np.dot(x, output_dense_weight.T) + output_dense_bias  # shape: [max_len, intermediate_size] (4, 768)
        return outputs

    
    # Multi-Head操作
    def multi_head(sele,x,attention_head_size,num_attention_heads):
        # hidden_size = 768  attention_head_size = 12 attention_head_size = 64
        max_len, hidden_size = x.shape
        x = x.reshape(max_len, num_attention_heads, attention_head_size)    # shape: (4, 12, 64)
        x = x.swapaxes(1, 0)    # shape:[num_attention_heads, max_len, attention_head_size] (12, 4, 64)
        return x
    
    # 整体流程
    def forward(self, x):
        # # --->
        # # 统计参数
        # num_token_embedding = 30522
        # num_segment_embedding = 2
        # num_position_embedding = 512
        # num_intermediate_size = 3072
        # #(30522+2+512)*768=23835648
        # num_embedding = (num_token_embedding+num_segment_embedding+num_position_embedding)*self.hidden_size
        # #12*(768*768/12*3*12+768*768)=28311552
        # num_multi_head= self.num_attention_heads*(self.hidden_size*self.hidden_size/self.num_attention_heads*3*self.num_attention_heads +self.hidden_size*self.hidden_size)
        # # 768*2+12*(768*2+768*2)=3840
        # num_layer = self.hidden_size*num_segment_embedding + self.num_attention_heads*(self.hidden_size*num_segment_embedding + self.hidden_size*num_segment_embedding )
        # # (768*3072+3072*768)=56623104
        # num_forward = self.num_attention_heads*(self.hidden_size*num_intermediate_size + self.hidden_size*num_intermediate_size)
        # #23835648+28311552+38400+56623104
        # total= num_embedding+num_multi_head+num_layer+num_forward
        
        # print("embedding层的参数为:\n Token_embedding: %d个, Segment_embedding: %d, Segment_embedding: %d\n 总计为: %d * %d = %d " %
        #       (num_token_embedding,num_segment_embedding,num_position_embedding,num_token_embedding+num_segment_embedding+num_position_embedding, self.hidden_size,num_embedding))
        # print("multi-head attention 三个权重矩阵拼接后经过一个线性变换, 每层参数为: %d * %d / %d * 3 * %d + %d * %d\n12层参数为: %d" %
        #       (self.hidden_size,self.hidden_size,self.num_attention_heads,self.num_attention_heads, self.hidden_size,self.hidden_size,num_multi_head))
        # print("layer normalization参数为 %d * %d + %d *(%d * %d + %d * %d) = %d"
        #       %(self.hidden_size,num_segment_embedding,self.num_attention_heads,self.hidden_size,num_segment_embedding,self.hidden_size,num_segment_embedding,num_layer))
        # print("feed forward层的参数为:\n %d * (%d * %d + %d * %d)= %d" %
        #       (self.num_attention_heads,self.hidden_size,num_intermediate_size,num_intermediate_size,self.hidden_size,num_forward))
        # print("总的参数 %d"% total)
        # #<-----
        
        x = self.emmbedding_forward(x)
        # 执行全部的transformer层计算
        seqence_output = self.all_transform_layer_forward(x)
        # 链接[cls] token的输出层
        pooler_output = self.pooler_out_layer(seqence_output[0])
        
        return seqence_output,pooler_output

# softmax
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)
# gelu激活函数
def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2/ math.pi) * (x + 0.044715 * np.power(x, 3))))

db = DiyBert(state_dict)
diy_sequence_output,diy_pooler_output = db.forward(x)
torch_sequence_output, torch_pooler_output = bert(torch_x)

print(diy_sequence_output)
print(torch_sequence_output)

"""
odict_keys的个数是:  23
embedding层的参数为:
 Token_embedding: 30522个, Segment_embedding: 2, Segment_embedding: 512
 总计为: 31036 * 768 = 23835648 
multi-head attention 三个权重矩阵拼接后经过一个线性变换, 每层参数为: 768 * 768 / 12 * 3 * 12 + 768 * 768
12层参数为: 28311552
layer normalization参数为 768 * 2 + 12 *(768 * 2 + 768 * 2) = 38400
feed forward层的参数为:
 12 * (768 * 3072 + 3072 * 768)= 56623104
总的参数 108808704
[[-0.12810704  0.0177492  -0.3158287  ...  0.08264183  0.04812796
   0.11650246]
 [-1.1661764   0.7968867  -0.85853887 ... -0.7675147  -0.0859234
  -0.02884368]
 [ 0.10833342  0.7716174   0.1706737  ... -0.921503   -0.35950267
   0.46555373]
 [-0.7294014  -0.23540282  0.00499353 ... -0.85136056 -0.17239603
   1.3457532 ]]
tensor([[[ 0.0169,  0.0160, -0.5747,  ..., -0.1830,  0.1512, -0.0556],
         [-1.2614,  0.6010, -0.7746,  ..., -0.5891, -0.2752,  0.1237],
         [-0.2093,  0.6627, -0.3288,  ..., -1.0358, -0.2932,  0.5032],
         [-0.7496, -0.1591, -0.4067,  ..., -0.6937,  0.1669,  1.3460]]],
       grad_fn=<NativeLayerNormBackward0>)
"""