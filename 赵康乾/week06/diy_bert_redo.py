import torch
import math
import numpy as np
from transformers import BertModel

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

'''
BertModel 需要实现embedding层, 多个encoder层(self_attention, feed_forward), 以及最后的pooler层

BertModel(
  (embeddings): BertEmbeddings(
    (word_embeddings): Embedding(vocab_size = 21128, 768, padding_idx=0)
    (position_embeddings): Embedding(max_position_embeddings = 512, 768)
    (token_type_embeddings): Embedding(2, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False), 只在训练时启用
  )
  (encoder): BertEncoder(
    (layer): ModuleList(
      (0-m): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False), 只在训练时启用
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False), 只在训练时启用
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False), 只在训练时启用
        )
      )
    )
  )
  (pooler): BertPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
  )
)

总参数量 num_parametes
embedding层, word_embeddings参数量 21128 * 768, position_embedding参数量 512 * 768, segment_embedding参数量 2 * 768, 层归一 768 + 768
encoder层中的self_attention层, q,k,v矩阵 3 * (768 * 768 + 768), 注意力线性层参数量 768 * 768 + 768, 层归一化参数量 768 + 768
encoder层中的feed_forward层, 放大的线性层 3072 * 768 + 3072, 缩小的线性层 768 * 3072 + 768, 层归一化参数量 768 + 768
pooler层, 768 * 768 + 768

23,849,334 + 1 * 7,088,336 + 590,592 = 31,528,272
'''


#读取预训练bert-base-chinese模型，把参数保存下来
bert = BertModel.from_pretrained("./bert-base-chinese", return_dict = False)
state_dict = bert.state_dict()

#模拟输入一个已经经过tokenizer的x，长度max_len是4，分词转化成了词表中的id
x = np.array([2450, 15486, 102, 2110])
torch_x = torch.LongTensor([x])


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis = -1, keepdims = True)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))


class DiyBert():
    #构造函数，按照config.jason构建参数，读取并载入bert-base-chinese模型参数
    def __init__(self, state_dict):
        self.num_attention_heads = 12      #每个transformer层中attention的头数
        self.hidden_size = 768             #每个词向量的维度
        self.max_position_embeddings = 512 #输入超过512就会被Tokenizer截断
        self.type_vocab_size = 2           #embeddings中segment的维度
        self.num_hidden_layers = 1         #transformer层的层数
        self.intermediate_size = 3072      #feed forward层的中间维度，先放大到4倍3072，再缩小到768
        self.load_weight(state_dict)
    
    def load_weight(self, state_dict):
        #加载预训练的bert-base-chinese模型参数
        #embeddings层
        self.word_embeddings = state_dict['embeddings.word_embeddings.weight'].numpy()             # [vocab_size, hidden_size]
        self.position_embeddings = state_dict['embeddings.position_embeddings.weight'].numpy()     # [max_position_embeddings, hidden_size]
        self.token_type_embeddings = state_dict['embeddings.token_type_embeddings.weight'].numpy() # [type_vocab_size, hidden_size]
        self.embeddings_layer_norm_weight = state_dict['embeddings.LayerNorm.weight'].numpy()      #embeddings层层归一化权重 [hidden_size]
        self.embeddings_layer_norm_bias = state_dict['embeddings.LayerNorm.bias'].numpy()          #embeddings层层归一化偏置 [hidden_size]

        #encoder层
        #一共有num_hidden_layers个encoder层，每层中有self_attention和feed_forward两个部分
        self.encoder_weights = []
        for i in range(self.num_hidden_layers):
            #self_attention层
            #q, k, v的权重与偏置
            q_w = state_dict['encoder.layer.%d.attention.self.query.weight' % i].numpy()   # query权重 [hidden_size, hidden_size]
            q_b = state_dict['encoder.layer.%d.attention.self.query.bias' % i].numpy()     # query偏置 [hidden_size]
            k_w = state_dict['encoder.layer.%d.attention.self.key.weight' % i].numpy()     # key权重 [hidden_size, hidden_size]
            k_b = state_dict['encoder.layer.%d.attention.self.key.bias' % i].numpy()       # key偏置 [hidden_size]
            v_w = state_dict['encoder.layer.%d.attention.self.value.weight' % i].numpy()   # value权重 [hidden_size, hidden_size]
            v_b = state_dict['encoder.layer.%d.attention.self.value.bias' % i].numpy()     # value偏置 [hidden_size]
            #Attention(Q,K,V)的结果先线性变换，再与残差连接后进行层归一化
            attention_output_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % i].numpy()     # 自注意力的输出结果进行线性变换的权重 [hidden_size, hidden_size]
            attention_output_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % i].numpy()         # 自注意力的输出结果进行线性变换的偏置 [hidden_size]
            attention_layer_norm_w = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i].numpy()  # 所有自注意力与残差链接后的层归一权重 [hidden_size]
            attention_layer_norm_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i].numpy()    # 所有自注意力与残差链接后的层归一偏置 [hidden_size]           
            
            #feed_forward层，先放大到4倍，再缩小回来
            #第一层放大的权重与偏置
            intermediate_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % i].numpy()    # 放大到4倍的权重 [4*hidden_size(intermediate_size), hidden_size]
            intermediate_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % i].numpy()        # 放大到4倍的偏置 [4*hidden_size(intermediate_size)]
            #第二层缩小的权重与偏置
            output_weight = state_dict["encoder.layer.%d.output.dense.weight" % i].numpy()                # 缩小回来的权重 [4*hidden_size(intermediate_size), hidden_size]
            output_bias = state_dict["encoder.layer.%d.output.dense.bias" % i].numpy()                    # 缩小回来的偏置 [hidden_size]
            #层归一化
            ff_layer_norm_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i].numpy()          # 层归一化权重 [hidden_size]
            ff_layer_norm_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i].numpy()            # 层归一化偏置 [hidden_size]
            
            #放进encoder_weights列表中
            self.encoder_weights.append([q_w, q_b, k_w, k_b, v_w, v_b, 
                                             attention_output_weight, attention_output_bias, 
                                             attention_layer_norm_w, attention_layer_norm_b, 
                                             intermediate_weight, intermediate_bias, 
                                             output_weight, output_bias, 
                                             ff_layer_norm_w, ff_layer_norm_b])
        
        #pooler层
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()  # shape:[hidden_size, hidden_size]
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()      # shape:[hidden_size]



    '''
    embedding层
    bert模型需要预先准备一个词表(vocab.txt), embedding层接受几句话作为输入, 
    将句子变成word_embeddings, position_embeddings, token_type_embeddings(segment_embeddings), 
    然后相加得到最终的embedding

    word_embeddings: embeddings.word_embeddings.weight中储存着vocab_size个向量[1, hidden_size],
    输入会在开头加上[CLS], 中断的位置加上[SEP], 每次输入的长度是max_len = 512, 不足的部分将会使用[PAD]来补齐
    根据分词在词表中的id来分配对应的向量
    position_embeddings: embeddings.position_embeddings.weight中储存着[max_position_embeddings, hidden_size]个向量, 按照分词在
    输入中的位置(0,1,2,3...)来分配对应的向量
    token_type_embeddings: 根据分词属于第几个句子来分配weight中的向量, config.json中的 "type_vocab_size": 2 表示输入会被分成两个句子，
    分词的序号会被分为0或1, [0,0,0,1,1]
    '''
    def embedding_forward(self, x):
        # 自定义get_embedding函数, 按照正确的index分配weights中的[hidden_size]向量
        # x.shape = [max_len]
        # word_embeddings是根据输入x的value选取向量，即根据分词在词表中的id来分配对应的向量
        word_embedding = self.get_embedding(self.word_embeddings, x)                                #shape: [max_len, hidden_size]
        # position_embeddings是根据分词在输入中的位置(0,1,2,3...)来分配对应的向量
        position_embeddings = self.get_embedding(self.position_embeddings, np.array(range(len(x)))) #shape: [max_len, hidden_size]
        # token_type_embeddings是根据根据分词属于第几个句子来分配weight中的向量，这里当作只有一个句子
        token_type_embeddings = self.get_embedding(self.token_type_embeddings, np.array([0] * len(x)))   #shape: [max_len, hidden_size], 注意np.zeros(len(x))生成的是float数组0.0
        # 相加并归一化得到embedding
        embedding = word_embedding + position_embeddings + token_type_embeddings
        embedding = self.layer_norm(embedding, self.embeddings_layer_norm_weight, self.embeddings_layer_norm_bias)  #shape: [max_len, hidden_size]
        return embedding
    
    #自定义的get_embedding函数, 按照正确的index分配weights中的[hidden_size]向量
    def get_embedding(self, embedding_matrix, x):
        return np.array([embedding_matrix[i] for i in x])

    #自定义的layer_norm函数, 对最后一维进行归一化
    def layer_norm(self, x, weight, bias):
        x = (x - np.mean(x, axis = -1, keepdims = True)) / np.std(x, axis = -1, keepdims=True)
        return x * weight + bias
    



    '''
    encoder层
    多个encoder层堆叠而成, 每个encoder层由两个子层组成, 一个是self-attention层, 另一个是feed_forward层
    '''
    '''
    self_attention层
    使用多头注意力,先根据q, k, v的 w 和 b 与 x 计算出Q, K, V, shape: [max_len, hidden_size], 每一行都是各个分词的q, k, v
    再按照多头数拆成多个q, k, v, shape: [max_len, num_attention_heads, head_size(hidden_size / num_attention_heads)]
    多头注意力中,每组头的q, k, v只和自己组的其他分词的q, k, v进行交互, 为了方便计算交换后两维, shape: [num_attention_heads, max_len, head_size]
    计算注意力得分：
    只看单个头, 对于每个分词(比如4个分词, max_len = 4),拿出q来和所有的k依次相乘,得到 4 个注意力权重,数学上就是 qk = q * k^T
    再对注意力得分/sqrt(d)后进行softmax, 得到每个分词对其他分词的注意力权重, [aij] shape:[4,4], 代表第i个分词对第j个分词的注意力权重 qk = softmax(qk / d^0.5)
    最后把注意力权重依次乘对应的vi, 得到每个分词的注意力得分
    也就是分词1的注意力得分是a11*v1 + a12*v2 + a13*v3 + a14*v4, 数学上就是矩阵乘法qk * v
    最后把多个头的注意力得分拼接起来, shape: [max_len, hidden_size]
    '''
    def self_attention(self, x, q_w, q_b, k_w, k_b, v_w, v_b, 
                       attention_output_weight, attention_output_bias, num_attention_heads, hidden_size):
        max_len = x.shape[0]
        # x.shape = [max_len, hidden_size]
        # q_w, k_w, v_w shape= [hidden_size, hidden_size]
        # q_b, k_b, v_b shape= [hidden_size]
        # attention_output_weight.shape = [hidden_size, hidden_size]
        # attention_output_bias.shape = [hidden_size]
        # num_attention_heads = 12
        # hidden_size = 768
        # head_size = hidden_size / num_attention_heads = 64
        q = np.dot(x, q_w.T) + q_b  #shape:[max_len, hidden_size]
        k = np.dot(x, k_w.T) + k_b  #shape:[max_len, hidden_size]
        v = np.dot(x, v_w.T) + v_b  #shape:[max_len, hidden_size]
 
        #把q, k, v按照num_attention_heads进行拆分并调整形状, shape: [num_attention_heads, max_len, head_size]
        head_size = int(hidden_size / num_attention_heads)
        q = q.reshape(max_len, num_attention_heads, head_size).transpose(1, 0, 2)
        k = k.reshape(max_len, num_attention_heads, head_size).transpose(1, 0, 2)
        v = v.reshape(max_len, num_attention_heads, head_size).transpose(1, 0, 2)

        #各个头内部各自计算注意力得分，Attention Score = softmax(QK^T / sqrt(d)) shape: [num_attention_heads, max_len, max_len]
        qk = np.matmul(q, k.swapaxes(1,2))
        qk /= np.sqrt(head_size)
        qk = softmax(qk)
        #计算Attention(q,k,v), shape:[num_attention_heads, max_len, hiddensize]
        qkv = np.matmul(qk, v)
        #把多个头的注意力得分拼接起来, shape: [max_len, hidden_size]
        qkv = qkv.swapaxes(0,1).reshape(max_len, hidden_size)
        #对拼接后的结果进行一次全连接层, shape: [max_len, hidden_size]
        qkv = np.dot(qkv, attention_output_weight.T) + attention_output_bias
        return qkv
    
    '''
    feed_forward层, 先从hidden_size维度映射到intermediate_size维度,放大4倍, 经过一层gelu, 再映射回hidden_size维度
    '''
    def feed_forward(self, x, intermediate_weight, intermediate_bias, output_weight, output_bias):
        # x.shape = [max_len, hidden_size]
        # intermediate_weight.shape = [intermediate_size, hidden_size]
        # intermediate_bias.shape = [intermediate_size]
        # output_weight.shape = [hidden_size, intermediate_size]
        # output_bias.shape = [hidden_size]
        # intermediate_size = 3072
        # hidden_size = 768
        x = np.dot(x, intermediate_weight.T) + intermediate_bias
        x = gelu(x)
        x = np.dot(x, output_weight.T) + output_bias
        return x
    
    # 定义一个encoder层
    def encoder(self, x, layer_index):
        weight_matrix = self.encoder_weights[layer_index]
        q_w, q_b, \
        k_w, k_b, \
        v_w, v_b, \
        attention_output_weight, attention_output_bias, \
        attention_layer_norm_w, attention_layer_norm_b, \
        intermediate_weight, intermediate_bias, \
        output_weight, output_bias, \
        ff_layer_norm_w, ff_layer_norm_b = weight_matrix

        #self_attention层
        attention_output = self.self_attention(x, q_w, q_b, k_w, k_b, v_w, v_b, 
                                               attention_output_weight, attention_output_bias, 
                                               self.num_attention_heads, self.hidden_size)
        #残差连接，拼接 x 和 attention_output后层归一化
        layer_norm_output = self.layer_norm(x + attention_output, attention_layer_norm_w, attention_layer_norm_b)
        #feed_forward层
        ff_output = self.feed_forward(layer_norm_output, intermediate_weight, intermediate_bias, output_weight, output_bias)
        #残差连接，拼接layer_norm_output 和 ff_output后层归一化
        output = self.layer_norm(layer_norm_output + ff_output, ff_layer_norm_w, ff_layer_norm_b)

        return output
    
    #按照定义的encoder数量(self.num_hidden_layers)堆叠encoder
    def encoder_stack(self, x):
        for i in range(self.num_hidden_layers):
            x = self.encoder(x, i)
        return x

    '''
    Pooler层
    由于所有的子任务都是基于【CLS】这个无语义token的embedding来进行的,
    所以最后一层Pooler层会对这一部分【CLS】的embedding再进行一层FFN,
    这样做的目的是一方面可以增加模型的复杂度, 提高表示能力, 
    另一方面, 全连接激活的输出形成的是一个固定长度的向量, 和输入序列长度无关, 这对于某些需要固定长度输入的模型是有益的。
    '''
    def pooler(self, x):
        x = np.dot(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
        output = np.tanh(x)
        return output

    
    #定义一个bert模型
    def forward(self,x):
        x = self.embedding_forward(x)
        sequence_output = self.encoder_stack(x)
        pooler_output = self.pooler(sequence_output[0])
        return sequence_output, pooler_output


#对比BertModel和DIYBertModel的输出
diybert = DiyBert(state_dict)
diy_sequence_output, diy_pooler_output = diybert.forward(x)
torch_sequence_output, torch_pooler_output = bert(torch_x)

print(diy_sequence_output)
print(torch_sequence_output)
