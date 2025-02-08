from transformers import BertModel
import torch
import numpy as np
import math
BERT_PATH = r"D:\ai\bert_model\bert-base-chinese" # bert-base-chinese 本地路径
bert_model = BertModel.from_pretrained(BERT_PATH)
BERT_MODEL_STATE_DICT =  bert_model.state_dict() # 获取 模型权重参数
print(BERT_MODEL_STATE_DICT.keys())

# 自定义bert模型
class Diy_bert_model:
    def params_size(self): # 参数个数计算
        num_layer =  self.num_layer # 中间层的数量
        hidden_size = self.hidden_size  #隐藏层大小
        vocab_size = self.vocab_size # 词表大小
        max_position_embeddings = self.max_position_embeddings #最大词数量
        # embedding 参数大小 = word_embedding + position_embennding + hidden_embedding + 一个layer_noram
        embedding_params_num = vocab_size*hidden_size + max_position_embeddings*hidden_size + 2*hidden_size + 2*hidden_size
        #attention 层参数大小 = k q v 三个 线性层 + 一个输出线性层 + 归一化参数
        attention_params_num = 4 * (hidden_size*hidden_size + 1*hidden_size) + (2*hidden_size)
        # feed forward 参数 两个线性层 + 一个归一化层  第一个线性层进行升维 到 4倍  第二个线性层降维到初始化维度
        feed_forward_params_num =(4*hidden_size*hidden_size +4*hidden_size )+ (4*hidden_size*hidden_size  +hidden_size) + 2*hidden_size
        # pool 层参数
        pool_num = hidden_size*hidden_size + 1*hidden_size
        return embedding_params_num + num_layer*(attention_params_num + feed_forward_params_num)+ pool_num


    def __init__(self,num_layer=1,num_attention_heads = 12,hidden_size=768,max_position_embeddings=512,vocab_size=21128 ):
        super(Diy_bert_model,self).__init__()
        self.num_layer = num_layer # 中间层的数量
        self.num_attention_heads = num_attention_heads # self-attention muil-head 拆分数量
        self.hidden_size = hidden_size #隐藏层数量
        self.vocab_size = vocab_size # 词表大小
        self.max_position_embeddings = max_position_embeddings #最大字符数量


        self.embedding_word_weight = BERT_MODEL_STATE_DICT['embeddings.word_embeddings.weight'].numpy()
        self.embedding_position_weight = BERT_MODEL_STATE_DICT['embeddings.position_embeddings.weight'].numpy()
        self.embedding_token_weight = BERT_MODEL_STATE_DICT['embeddings.token_type_embeddings.weight'].numpy()

        self.embedding_layerNorm_weight = BERT_MODEL_STATE_DICT['embeddings.LayerNorm.weight'].numpy()
        self.embeddings_layerNorm_bias = BERT_MODEL_STATE_DICT['embeddings.LayerNorm.bias'].numpy()
        self.init_layer_params()




    def init_layer_params(self):
        self.encoder_layer = [0]*self.num_layer
        for i in range(self.num_layer):
            self.encoder_layer[i] = {
                'attention_query_weight':BERT_MODEL_STATE_DICT['encoder.layer.%d.attention.self.query.weight'%i].numpy() ,
                'attention_query_bias':BERT_MODEL_STATE_DICT['encoder.layer.%d.attention.self.query.bias'%i].numpy() ,
                'attention_key_weight':BERT_MODEL_STATE_DICT['encoder.layer.%d.attention.self.key.weight'%i].numpy() ,
                'attention_key_bias':BERT_MODEL_STATE_DICT['encoder.layer.%d.attention.self.key.bias'%i].numpy() ,
                'attention_value_weight':BERT_MODEL_STATE_DICT['encoder.layer.%d.attention.self.value.weight'%i].numpy() ,
                'attention_value_bias':BERT_MODEL_STATE_DICT['encoder.layer.%d.attention.self.value.bias'%i].numpy() ,
                'attention_output_dense_weight' : BERT_MODEL_STATE_DICT['encoder.layer.%d.attention.output.dense.weight'%i].numpy() ,
                'attention_output_dense_bias' : BERT_MODEL_STATE_DICT['encoder.layer.%d.attention.output.dense.bias'%i].numpy() ,
                'attention_output_layerNorm_weight':BERT_MODEL_STATE_DICT['encoder.layer.%d.attention.output.LayerNorm.weight'%i].numpy() ,
                'attention_output_layerNorm_bias':BERT_MODEL_STATE_DICT['encoder.layer.%d.attention.output.LayerNorm.bias'%i].numpy() ,
                'intermediate.dense.weight' : BERT_MODEL_STATE_DICT['encoder.layer.%d.intermediate.dense.weight'%i].numpy() ,
                'intermediate.dense.bias' : BERT_MODEL_STATE_DICT['encoder.layer.%d.intermediate.dense.bias'%i].numpy() ,
                'output.dense.weight' : BERT_MODEL_STATE_DICT['encoder.layer.%d.output.dense.weight'%i].numpy() ,
                'output.dense.bias' : BERT_MODEL_STATE_DICT['encoder.layer.%d.output.dense.bias'%i].numpy() ,
                'output_layerNorm_weight':BERT_MODEL_STATE_DICT['encoder.layer.%d.output.LayerNorm.weight'%i].numpy() ,
                'output_layerNorm_bias':BERT_MODEL_STATE_DICT['encoder.layer.%d.output.LayerNorm.bias'%i].numpy() ,
            }

    def forward(self,X):
        X = self.embedding_word_weight[X] # 获取词向量 shape = batch_size * length * 768 
        length = X.shape[0]
        X = X + self.embedding_position_weight[0:length]  #加 position embedding 

        X = X + [self.embedding_token_weight[0]] * length  # 单行输入 token type 取脚标为 0 
    

        X = self.layer_norm(self.embedding_layerNorm_weight,self.embeddings_layerNorm_bias,X)  # embedding 加和后过  归一化层
        for i in range(self.num_layer):
            layer = self.encoder_layer[i]
            X = self.encoder(layer,X)
        
        return X
    
    def layer_norm(self,w,b,x):  
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        x = x * w + b
        return x
   
    def encoder(self,layer,X):
        Q = np.dot(X,layer['attention_query_weight'].T)+layer['attention_query_bias']
        K = np.dot( X,layer['attention_key_weight'].T) +layer['attention_key_bias'] 
        V =np.dot( X,layer['attention_value_weight'].T) +layer['attention_value_bias'] # length*768  
        # 计算公式 Q*k/  self-attention 计算方式
        Q_arr = np.hsplit(Q,self.num_attention_heads)  #  length * 64 (768/12 )
        K_arr = np.hsplit(K,self.num_attention_heads)
        V_arr = np.hsplit(V,self.num_attention_heads)
        r = [0]*self.num_attention_heads
        for i in range(self.num_attention_heads):
           r[i] = np.dot( Q_arr[i] , K_arr[i].T )               # length*64  * 64*length = length*length 
           r[i] = softMax(r[i]/np.sqrt(64))
           r[i] = np.dot(  r[i] ,  V_arr[i])                   # length*length   length*64 = length*64
        output = np.hstack(r)                                        # 合并 计算结果   length * 768
        output = np.dot(output, layer['attention_output_dense_weight'].T) + layer['attention_output_dense_bias'] # 过一个线性层
        X = self.layer_norm(layer['attention_output_layerNorm_weight'] ,layer['attention_output_layerNorm_bias'] ,X + output) # 残差机制 + 归一化 

        # feed forward 计算
        output = np.dot(X,layer['intermediate.dense.weight'].T)+layer['intermediate.dense.bias']
        output = gelu(output) # 激活函数
        output = np.dot(output,layer['output.dense.weight'].T)+layer['output.dense.bias']
        X = self.layer_norm(layer['output_layerNorm_weight'] ,layer['output_layerNorm_bias'] ,X + output) # 残差机制 + 归一化 
        return X 





def softMax(X):
    return  np.exp(X)/np.sum(np.exp(X),axis=-1,keepdims=True)

#gelu激活函数
def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))

X = [2450, 15486, 102, 2110]
X_tensor = torch.tensor([[2450, 15486, 102, 2110]])

diy_bert_model =  Diy_bert_model(num_layer=12)
print(diy_bert_model.forward(X))
print(bert_model(X_tensor)['last_hidden_state'])

print("计算参数数量:%s"%diy_bert_model.params_size())


actual_size = 0
for key in BERT_MODEL_STATE_DICT.keys():
    actual_size +=np.size(BERT_MODEL_STATE_DICT[key].numpy())
print("实际参数数量:%s" %actual_size)


