import torch
import math
import numpy as np
from transformers import BertModel

#   return_dict (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
# 默认为False;
path = r"I:\BaiduNetdiskDownload\八斗精品课\第六周 语言模型\bert-base-chinese\bert-base-chinese"
bert = BertModel.from_pretrained(path,return_dict=False)
state_dict = bert.state_dict()
# 将模型调整到执行模式,其中有丢弃层时将不起作用
bert.eval()
x = np.array([2450, 15486, 102, 2110]) # 假象成4个字
torch_x = torch.LongTensor([x]) #pytorch的形式输入
seqence_output, pooler_output = bert(torch_x)
print(seqence_output.shape,pooler_output.shape)

print(bert.state_dict().keys())

#softmax归一化
# [ 1 ,  2 , 3 ]
#  softmax
# [ e1/(e1+e2+e3), e2/(e1+e2+e3), e3/(e1+e2+e3)]
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=-1,keepdims=True)

#gelu激活函数
#GELU(x)=0.5x[1+tanh(√(π/2)(x+0.047715x³))]
def gelu(x):
    return 0.5*x*(1+np.tanh(math.sqrt(2/math.pi)*(x+0.044715*np.power(x,3))))

class BertWork:
    #将预训练好的整个权重字典输入进来
    def __init__(self,state_dict):
        self.num_attention_heads = 12 #多头multi-head的数量
        self.hidden_size = 768 # 隐藏层大小
        self.num_layers = 1 # transformer 的层数
        self.load_weights(state_dict)

    def load_weights(self,state_dict):
        #embedding部分(vocab.size(21128) x 768)
        self.word_embedding = state_dict["embeddings.word_embeddings.weight"].numpy()
        #Position Embedding层,带入语序信息(BERT的base版本位置信息最大512,所以当时超过512个字就失效了)
        self.postion_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()
        #Token Type Embedding（也称为Segment Embedding）(2x768)
        self.tokey_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()
        # 归一化的k
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()
        self.transformer_weights = []
        #transformer部分,有多层,原始BERT是12层
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
            self.transformer_weights.append([q_w,q_b,k_w,k_b,v_w,v_b,attention_output_weight,attention_output_bias,attention_layer_norm_w,attention_layer_norm_b,
                                             intermediate_weight,intermediate_bias,output_weight,output_bias,ff_layer_norm_w,ff_layer_norm_b])

        #pooler 层
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()
    #bert embedding,使用3层叠加,再经过一个Layer norm归一化层
    def embedding_forward(self,x):
        #x.shape = [max_len] 4
        we = self.get_embedding(self.word_embedding,x) #shape:[max_len,hidden_size] [4,768]
        #position embedding的输入[0,1,2,3]
        pe = self.get_embedding(self.postion_embeddings,np.array(list(range(len(x))))) #shape:[max_len,hidden_size] [4,768]
        #token type embedding,单输入的情况下为[0,0,0,0] 也称为Segment Embedding
        te = self.get_embedding(self.tokey_type_embeddings,np.array([0]*len(x))) #shape:[max_len,hidden_size] [4,768]
        embedding = we+pe+te
        #加和后过一个归一化层
        embedding = self.layer_norm(embedding,self.embeddings_layer_norm_weight,self.embeddings_layer_norm_bias) # shpae: [max_len, hidden_size] [4,768]
        return embedding

    #embdedding层实际上相当于按index索引,或者理解为onehot输入乘以embedding矩阵
    def get_embedding(self, embedding_matrix, x):
        return np.array([embedding_matrix[index] for index in x])

    #归一化层
    def layer_norm(self,x,w,b):
        x = (x - np.mean(x,axis=1,keepdims=True))/ np.std(x,axis=1,keepdims=True)
        x = x*w+b
        return x
    #执行全部的transformer层的计算
    def all_transformer_layer_forward(self,x):
        for i in range(self.num_layers):
            x = self.single_transformer_layer_forward(x,i)
        return x

    #执行单层transformer层的计算
    def single_transformer_layer_forward(self,x,layer_index):
        #取出该层的参数,在实际中,这些参数都是随机初始化,之后进行预训练
        weights = self.transformer_weights[layer_index]
        q_w,q_b, \
        k_w,k_b, \
        v_w,v_b, \
        attention_output_weight,attention_output_bias, \
        attention_layer_norm_w,attention_layer_norm_b, \
        intermediate_weight,intermediate_bias, \
        output_weight,output_bias, \
        ff_layer_norm_w,ff_layer_norm_b, \
         = weights
        #self attention层
        attention_output = self.self_attention(x,q_w,q_b,k_w,k_b,v_w,v_b,
                                               attention_output_weight,attention_output_bias,self.num_attention_heads,self.hidden_size)
        #bn层,并使用了残差机制
        x = self.layer_norm(x + attention_output,attention_layer_norm_w,attention_layer_norm_b)
        #feed forward层
        feed_forward_x = self.feed_forward(x,intermediate_weight, intermediate_bias, output_weight, output_bias)
        #bn层,并使用了残差机制
        x = self.layer_norm(x+feed_forward_x,ff_layer_norm_w,ff_layer_norm_b)
        return x


    # self attention的计算
    def self_attention(self,x,q_w,q_b,k_w,k_b,v_w,v_b,
                       attention_output_weight,attention_output_bias,num_attention_heads,hidden_size):
        #x.shape = max_len * hidden_size
        #q_w,k_w,v_w shape = hidden_size * hidden_size
        #q_b,k_b,v_b shape = hidden_size
        q = np.dot(x,q_w.T) + q_b # shape: [max_len,hidden_size] W*X+B 线性层
        k = np.dot(x,k_w.T) + k_b # shape: [max_len,hidden_size]
        v = np.dot(x,v_w.T) + v_b # shape: [max_len,hidden_size]
        attention_head_size = int(hidden_size / num_attention_heads)
        #q.shape = num_attention_heads,max_len,attention_head_size
        q = self.transpose_for_scores(q,attention_head_size, num_attention_heads)
        #k.shape = num_attention_heads,max_len,attention_head_size
        k = self.transpose_for_scores(k,attention_head_size,num_attention_heads)
        #v.shape = num_attention_heads,max_len,attention_head_size
        v = self.transpose_for_scores(v,attention_head_size,num_attention_heads)
        #qk.shape = num_attention_heads,max_len,max_len
        qk = np.matmul(q,k.swapaxes(1,2))
        qk /= np.sqrt(attention_head_size)
        qk = softmax(qk)
        #qkv.shape = num_attention_heads,max_len,attention_head_size
        qkv = np.matmul(qk,v)
        # qkv.shape = max_len_hidden_size
        qkv = qkv.swapaxes(0,1).reshape(-1,hidden_size)
        # attention.shape = max_len,hiden_size
        attention = np.dot(qkv,attention_output_weight.T) + attention_output_bias
        return attention

    # 多头机制 multi-head
    def transpose_for_scores(self,x,attention_head_size,num_attention_heads):
        # hidden_size = 768, num_attention_heads = 12, attention_head_size = 64
        max_len,hidden_size = x.shape
        x = x.reshape(max_len,num_attention_heads,attention_head_size)
        x = x.swapaxes(1,0) # a.shape(2，3，4）,a.swapaxes()
                            # 是将n维数组中两个维度进行调换，其中x，y的值为a.shape值（2，3，4）元组中的索引值
        return x
    # 前馈网络的计算 feed_forward: Liner(gelu(Liner(x)))
    def feed_forward(self,x,intermediate_weight,intermediate_bias,output_weight,output_bias):
        # output.shape:[max_len,intermediate_size]
        x = np.dot(x,intermediate_weight.T) + intermediate_bias
        x = gelu(x)
        #output.shape: [max_len,hidden_size]
        x = np.dot(x,output_weight.T) + output_bias
        return x
    #链接[cls] token的输出层
    def pooler_output_layer(self,x):
        x = np.dot(x,self.pooler_dense_weight.T) + self.pooler_dense_bias
        x = np.tanh(x)
        return x
    #最终输出
    def forworad(self,x):
        x = self.embedding_forward(x)
        seqence_output = self.all_transformer_layer_forward(x)
        pooler_output = self.pooler_output_layer(seqence_output[0])
        return seqence_output,pooler_output

#自制
db = BertWork(state_dict)
diy_sequence_output,diy_pooler_output = db.forworad(x)
#torch
torch_sequence_output,torch_pooler_output = bert(torch_x)
print(diy_sequence_output)
print("---------------------")
print(torch_sequence_output)

#12 * single_transformer_layer_forward

#embedding 层 vocab.size x hidden_size + hidden_size x 512 + hidden_size x 2

#word_embeddings embedding部分(vocab.size x hidden_size)
#postion_embeddings hidden_size x 512
#tokey_type_embeddings hidden_size x 2

# self_attention 层 Liner(Attention(Q,K,V))
# multi-head 机制中, 768 分12 头, 每格为64,即单头 为 64 x hidden_size
# q_w ,q_b, 12(64 x hidden_size+1 x hidden_size) Q[L,64],K[L,64].T=[L,L]*V[L,64]=12 x z[L,64] = Z[L,768]
# k_w ,k_b, 12(64 x hidden_size+1 x hidden_size)
# v_w ,v_b, 12(64 x hidden_size+1 x hidden_size)
# attention_output_weight ,attention_output_bias     hidden_size x hidden_size + hidden_size

#bn层 layer_norm 有一个Liner  hidden_size x hidden_size + hidden_size
# attention_layer_norm_w,attention_layer_norm_b, hidden_size x hidden_size + hidden_size

#feed forward层 Liner(gelu(Liner(x))) => 线性层->gelu激活->线性层 => 2(hidden_size x hidden_size + hidden_size)
# intermediate_weight,intermediate_bias, hidden_size x hidden_size + hidden_size
# output_weight,output_bias, hidden_size x hidden_size + hidden_size

#bn层 layer_norm 有一个Liner hidden_size x hidden_size + hidden_size
# ff_layer_norm_w,ff_layer_norm_b, hidden_size x hidden_size + hidden_size

#embedding层 vocab.size x hidden_size + hidden_size x 512 + hidden_size x 2
#self_attention层 3x12(64 x hidden_size+1 x hidden_size)
#bn层1 hidden_size x hidden_size + hidden_size
#feed forward层 2(hidden_size x hidden_size + hidden_size)
#bn层2 hidden_size x hidden_size + hidden_size

# vocab.size * hidden_size + hidden_size * max_position_embeddings + 2 * hidden_size + num_attention_heads x 3(64 x hidden_size+1 x hidden_size) + 4(hidden_size x hidden_size + hidden_size)
# embedding层(word_embeddings + postion_embeddings + tokey_type_embeddings)
# +self_attention层(num_attention_heads*(3(64 x hidden_size+1 x hidden_size))+Liner(hidden_size x hidden_size + hidden_size))
# +bn层1.Liner(hidden_size x hidden_size + hidden_size)
# feed forward层Liner(gelu(Liner(x)))(2(hidden_size x hidden_size + hidden_size))
# +bn层2.Liner(hidden_size x hidden_size + hidden_size)

# num_parameter = num_layers(vocab.size * hidden_size + max_position_embeddings * hidden_size + 2 * hidden_size + num_attention_heads * 3(64 * hidden_size+1 * hidden_size)+5(hidden_size x hidden_size + hidden_size))
# max_position_embeddings = 512
# vocab.size = 21128
# num_attention_heads = 12

# num_parameter_formula =num_layers*( 21128 * hidden_size + 512 * hidden_size + 2 * hidden_size + 12 * 3*(64 * hidden_size+1 * hidden_size)+5(hidden_size * hidden_size + hidden_size))
#                       =num_layers*( 23987 * hidden_size + 5*hidden_size²)






