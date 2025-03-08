import torch
import math
import numpy as np
from transformers import BertModel
import warnings
warnings.filterwarnings('ignore')

'''

通过手动矩阵运算实现Bert结构
模型文件下载 https://huggingface.co/models

'''

bert = BertModel.from_pretrained(r"E:/AIGC/NLP算法/【6】语言模型/bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
bert.eval()
x = np.array([2450, 15486, 102, 2110])   #假想成4个字的句子
torch_x = torch.LongTensor([x])          #pytorch形式输入
seqence_output, pooler_output = bert(torch_x)
print(seqence_output.shape, pooler_output.shape)
# print(seqence_output, pooler_output)

# print(bert.state_dict().keys())  #查看所有的权值矩阵名称

#softmax归一化
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)

#gelu激活函数
def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))

class DiyBert:
    #将预训练好的整个权重字典输入进来
    def __init__(self, state_dict):
        self.num_attention_heads = 12  # 注意力头
        self.hidden_size = 768         # 隐藏层
        self.num_layers = 12            # 注意这里的层数要跟预训练config.json文件中的模型层数一致  12层Transformer编码器
        self.load_weights(state_dict)

    def load_weights(self, state_dict):
        #embedding部分
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy()              # 获取BERT的词嵌入矩阵 (vocab_size, hidden_size)
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()      # 获取位置嵌入矩阵 (max_position, hidden_size)
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()  # 获取token类型编码（区分句子） (2, hidden_size)

        # 获取嵌入层的LayerNorm权重和偏置，用于对词嵌入、位置嵌入和token类型嵌入的结果进行归一化
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()
        self.transformer_weights = []
        #transformer部分，有多层
        for i in range(self.num_layers):
            q_w = state_dict[f"encoder.layer.{i}.attention.self.query.weight"].numpy()
            q_b = state_dict[f"encoder.layer.{i}.attention.self.query.bias"].numpy()
            k_w = state_dict[f"encoder.layer.{i}.attention.self.key.weight"].numpy()
            k_b = state_dict[f"encoder.layer.{i}.attention.self.key.bias"].numpy()
            v_w = state_dict[f"encoder.layer.{i}.attention.self.value.weight"].numpy()
            v_b = state_dict[f"encoder.layer.{i}.attention.self.value.bias"].numpy()
            attention_output_weight = state_dict[f"encoder.layer.{i}.attention.output.dense.weight"].numpy()
            attention_output_bias = state_dict[f"encoder.layer.{i}.attention.output.dense.bias"].numpy()
            attention_layer_norm_w = state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.weight"].numpy()
            attention_layer_norm_b = state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.bias"].numpy()
            intermediate_weight = state_dict[f"encoder.layer.{i}.intermediate.dense.weight"].numpy()
            intermediate_bias = state_dict[f"encoder.layer.{i}.intermediate.dense.bias"].numpy()
            output_weight = state_dict[f"encoder.layer.{i}.output.dense.weight"].numpy()
            output_bias = state_dict[f"encoder.layer.{i}.output.dense.bias"].numpy()
            ff_layer_norm_w = state_dict[f"encoder.layer.{i}.output.LayerNorm.weight"].numpy()
            ff_layer_norm_b = state_dict[f"encoder.layer.{i}.output.LayerNorm.bias"].numpy()
            self.transformer_weights.append([q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                                            attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias,
                                            output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])

        # Pooler层的权重和偏置
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()

        '''参数输出'''
        print(f"vocab_size: ", self.word_embeddings.shape[0])               # 打印词汇表大小 (vocab_size)
        print(f"max_position: ", self.position_embeddings.shape[0])         # 打印最大位置长度 （max_position）
        print(f"token_size: ", self.token_type_embeddings.shape[0])         # 句子类型
        print(f"intermediate_weight: ", intermediate_weight.shape[0])       # 前馈网络中间层维度
        print(f"pooler_dense_weight: ", self.pooler_dense_weight.shape[0])  # pooler层的权重


    '''embedding层'''
    def embedding_forward(self, x):
        # 词嵌入
        we = self.get_embedding(self.word_embeddings, x)  # shpae: [max_len, hidden_size]
        # 位置嵌入
        pe = self.get_embedding(self.position_embeddings, np.array(list(range(len(x)))))  # shpae: [max_len, hidden_size]
        # 句子类型嵌入
        te = self.get_embedding(self.token_type_embeddings, np.array([0] * len(x)))  # shpae: [max_len, hidden_size]
        embedding = we + pe + te
        # 加和后 -> 归一化层
        embedding = self.layer_norm(embedding, self.embeddings_layer_norm_weight, self.embeddings_layer_norm_bias)  # shpae: [max_len, hidden_size]
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

        #self-attention层
        attention_output = self.self_attention(
            x, q_w, q_b, k_w, k_b, v_w, v_b, 
            attention_output_weight, attention_output_bias,
            self.num_attention_heads, self.hidden_size
        )

        #bn层，并使用了残差机制
        x = self.layer_norm(x + attention_output, attention_layer_norm_w, attention_layer_norm_b)
        #feed forward层
        feed_forward_x = self.feed_forward(x,
                              intermediate_weight, intermediate_bias,
                              output_weight, output_bias)
        #bn层，并使用了残差机制
        x = self.layer_norm(x + feed_forward_x, ff_layer_norm_w, ff_layer_norm_b)
        return x

    # self attention的计算
    def self_attention(self, x, q_w, q_b, k_w, k_b, v_w, v_b,
            attention_output_weight, attention_output_bias,
            num_attention_heads, hidden_size
        ):
        # x.shape = max_len * hidden_size
        # q_w, k_w, v_w  shape = hidden_size * hidden_size
        # q_b, k_b, v_b  shape = hidden_size
        # 计算查询（Q）、键（K）、值（V）
        q = np.dot(x, q_w.T) + q_b  # shape: [max_len, hidden_size]      W * X + B lINER
        k = np.dot(x, k_w.T) + k_b  # shpae: [max_len, hidden_size]
        v = np.dot(x, v_w.T) + v_b  # shpae: [max_len, hidden_size]

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
        x = gelu(x)
        # output shpae: [max_len, hidden_size]
        x = np.dot(x, output_weight.T) + output_bias
        return x

    #归一化层
    def layer_norm(self, x, w, b):
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        x = x * w + b
        return x

    #链接[cls] token的输出层
    def pooler_output_layer(self, x):
        x = np.dot(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
        x = np.tanh(x)
        return x

    #最终输出
    def forward(self, x):
        x = self.embedding_forward(x)
        sequence_output = self.all_transformer_layer_forward(x)
        pooler_output = self.pooler_output_layer(sequence_output[0])
        return sequence_output, pooler_output

    def count_total_parameters(self):
            total_params = 0
            # 计算Embedding层参数数量
            total_params += np.prod(self.word_embeddings.shape) # hidden_size*vocab_size
            # print(self.word_embeddings.shape[0])
            # print(self.word_embeddings.shape[1])
            # print(total_params)
            total_params += np.prod(self.position_embeddings.shape) # hidden_size*max_position_embeddings
            total_params += np.prod(self.token_type_embeddings.shape) # hidden_size*type_vocab_size
            total_params += np.prod(self.embeddings_layer_norm_weight.shape) # hidden_size
            total_params += np.prod(self.embeddings_layer_norm_bias.shape) # hidden_size

            num_embedded = total_params
            print(f"embedded层参数量: ", num_embedded)

            # 计算Transformer层参数数量
            for weights in self.transformer_weights:
                q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias, \
                attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias, \
                output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b = weights
                total_params += np.prod(q_w.shape) # hidden_size*hidden_size
                total_params += np.prod(q_b.shape) # hidden_size
                total_params += np.prod(k_w.shape) # hidden_size*hidden_size
                total_params += np.prod(k_b.shape) # hidden_size
                total_params += np.prod(v_w.shape) # hidden_size*hidden_size
                total_params += np.prod(v_b.shape) # hidden_size
                total_params += np.prod(attention_output_weight.shape) # hidden_size*hidden_size
                total_params += np.prod(attention_output_bias.shape) # hidden_size
                total_params += np.prod(attention_layer_norm_w.shape) # hidden_size
                total_params += np.prod(attention_layer_norm_b.shape) # hidden_size
                total_params += np.prod(intermediate_weight.shape) # intermediate_size*hidden_size
                total_params += np.prod(intermediate_bias.shape) # intermediate_size
                total_params += np.prod(output_weight.shape) # hidden_size*intermediate_size
                total_params += np.prod(output_bias.shape) # hidden_size
                total_params += np.prod(ff_layer_norm_w.shape) # hidden_size
                total_params += np.prod(ff_layer_norm_b.shape) # hidden_size
                print(total_params - num_embedded)
            
            num_transformer = total_params - num_embedded
            print(f"transformer层参数量: ", num_transformer)

            # 计算Pooler层参数数量
            total_params += np.prod(self.pooler_dense_weight.shape) # hidden_size*hidden_size
            total_params += np.prod(self.pooler_dense_bias.shape) # hidden_size

            num_pooler = total_params - num_embedded - num_transformer
            print(f"pooler层参数量: ", num_pooler)

            return total_params

#自制
db = DiyBert(state_dict)
diy_sequence_output, diy_pooler_output = db.forward(x)
#torch
torch_sequence_output, torch_pooler_output = bert(torch_x)

# print(diy_sequence_output)
# print(torch_sequence_output)

total_params = db.count_total_parameters()

print("DiyBert总的参数大小:", total_params)

'''手动计算bert的参数量解析'''
'''
1. Embedded层
    1.1 词嵌入矩阵 word_embeddings
        [vocab_size, hidden_size] -> [21128, 768]
        参数量: 21128 * 768 = 16226304
    1.2 位置嵌入矩阵 position_embeddings
        [max_position, hidden_size] -> [512, 768]
        参数量: 512 * 768 = 393216
    1.3 句子类型嵌入矩阵 token_type_embeddings
        [token_size, hidden_size] -> [2, 768]
        参数量: 2 * 768 = 1536
    1.4 嵌入层的LayerNorm参数
        权重weight: 1 * 768
        偏置bias:   1 * 768 
        参数量: 2 * 768 = 1536
    总计: 16226304 + 393216 + 1536 + 1536 = 16622592
2. transformer层 BERT-Base通常有12层(已修改代码)
    2.1 self-attention 自注意力机制
        2.1.1 Q, K, V 权重矩阵
            权重3 * [hidden_size, hidden_size] -> 3 * [768, 768]
            偏置 3 * 768 
            参数量: 3 * 768 * 768 + 3 * 768= 1771776
        2.1.2 Attention 输出矩阵
            权重[hidden_size, hidden_size] -> [768, 768]
            偏置 768
            参数量: 768 * 768 + 768 = 590592
        2.1.3 Attention 层的 LayerNorm
            权重weight: 1 * 768
            偏置bias:   1 * 768 
            参数量: 2 * 768 = 1536
    2.2 Feed-forward 层 前馈网络
        2.2.1 中间层
            权重[hidden_size, intermediate_size] -> [768, 3072]
            偏置 3072
            参数量: 768 * 3072 + 3072 = 2362368
        2.2.2 输出层
            权重[intermediate_size, hidden_size] -> [3072, 768]
            偏置 768
            参数量: 768 * 3072 + 768 = 2360064
        2.2.3 Feed-forward 层的 LayerNorm
            权重weight: 1 * 768
            偏置bias:   1 * 768 
            参数量: 2 * 768 = 1536
    每层的参数量: 1771776 + 590592 + 1536 + 2360064 + 2362368 + 1536 = 7087872
    12层的参数量: 7087872 * 12 = 85054464
3. Pooler 层
    3.1 Pooler 的 dense 权重矩阵
        [hidden_size, hidden_size] -> [768, 768]
        参数量: 768 * 768 = 589824
    3.2 Pooler 的 bias
        hidden_size -> 768
    总计: 589,824 + 768 = 590592

总参数量: 16622592 + 85054464 + 590592 = 102267648
'''
