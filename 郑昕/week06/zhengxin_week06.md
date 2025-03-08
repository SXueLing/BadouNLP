# 计算Bert参数量。
## Bert模型的组成
1. 嵌入层 （Embedding Layer）
   - 词嵌入 （Token Embeddings）
   - 分段嵌入（Segment Embeddings）
   - 位置嵌入（Position Embeddings）
2. 编码器 （Encoder）
3. 池化层 （Pooling Layer）

## 参数量计算的主要步骤
1. 计算嵌入层的参数量
   - 计算词嵌入层的参数量
     - 假设词汇表大小为  V ，嵌入维度为  d_{\text{model}} ，词嵌入的参数量为： V \times d_{\text{model}}
   - 计算分段嵌入层的参数量、
     - 假设最大序列长度为  L ，嵌入维度为  d_{\text{model}} ，位置嵌入的参数量为： L \times d_{\text{model}}
   - 计算位置嵌入层的参数量
     - 分段嵌入通常有两类（句子 A 和句子 B），嵌入维度为  d_{\text{model}} ，参数量为： 2 \times d_{\text{model}}
   - 总嵌入层的参数量
     -\text{嵌入层参数量} = V \times d_{\text{model}} + L \times d_{\text{model}} + 2 \times d_{\text{model}}

2. 计算编码器的参数量
   - 多头注意力机制（Multi-Head Attention）
     - 	每个注意力头有三个权重矩阵：查询 ( Q )、键 ( K ) 和值 ( V )，以及输出权重矩阵。
     - 假设注意力头数为  h ，每个头的维度为  d_k = d_{\text{model}} / h ，则：
       - Q, K, V  的参数量分别为：d_{\text{model}} \times d_k \quad (\text{每个头}) \quad \rightarrow \quad h \cdot d_{\text{model}} \cdot d_k
       - 多头注意力的输出权重矩阵为： d_{\text{model}} \times d_{\text{model}}
       - 总多头注意力参数量为： 3 \cdot h \cdot d_{\text{model}} \cdot d_k + d_{\text{model}} \cdot d_{\text{model}}

   - 前馈神经网络（Feed-Forward Neural Network）
     - 前向网络包括两个全连接层，第一层将  d_{\text{model}}  映射到  d_{\text{ffn}} ，第二层将  d_{\text{ffn}}  映射回  d_{\text{model}} ： d_{\text{model}} \times d_{\text{ffn}} + d_{\text{ffn}} \times d_{\text{model}}
        - 总前向网络参数量为： 2 \cdot d_{\text{model}} \cdot d_{\text{ffn}}
   - 层归一化（Layer Normalization）：
	 - 每层有两个归一化模块，每个归一化模块需要两个参数（缩放和偏移），总参数量为： 2 \cdot d_{\text{model}} \quad (\text{每层})
   - 编码器的参数量
     - \text{编码器参数量} = 3 \cdot h \cdot d_{\text{model}} \cdot d_k + d_{\text{model}} \cdot d_{\text{model}} + 2 \cdot d_{\text{model}} \cdot d_{\text{ffn}} + 2 \cdot d_{\text{model}} \quad (\text{每层})
3. 计算池化层的参数量
   - 池化层是一个全连接层，将 [CLS] 的输出变换为固定大小的句子向量，参数量为： d_{\text{model}} \times d_{\text{model}}

示例：Bert-base
参数设置：
- V = 30,000 
- L = 512 
- d_{\text{model}} = 768 
- h = 12 
- d_k = d_{\text{model}} / h = 64 
- d_{\text{ffn}} = 3072 
- N = 12 

1. 嵌入层参数量
\text{嵌入层参数量} = 30,000 \cdot 768 + 512 \cdot 768 + 2 \cdot 768 = 23,685,120
2. 编码器参数量(单层)
\text{单层编码器参数量} = 3 \cdot 12 \cdot 768 \cdot 64 + 768^2 + 2 \cdot 768 \cdot 3072 + 2 \cdot 768
= 1,769,472 + 589,824 + 4,718,592 + 1,536 = 7,079,424
3. 总编码器参数量
\text{总编码器参数量} = 12 \cdot 7,079,424 = 84,952,608
4. 池化层参数量
\text{池化层参数量} = 768^2 = 589,824
5. 总参数量
\text{总参数量} = 23,685,120 + 84,952,608 + 589,824 = 109,227,552

# 总结： 对于任意 BERT 模型，总参数量为：
\text{总参数量} = V \cdot d_{\text{model}} + L \cdot d_{\text{model}} + 2 \cdot d_{\text{model}} + N \cdot \left( 3 \cdot h \cdot d_{\text{model}} \cdot \frac{d_{\text{model}}}{h} + d_{\text{model}}^2 + 2 \cdot d_{\text{model}} \cdot d_{\text{ffn}} + 2 \cdot d_{\text{model}} \right) + d_{\text{model}}^2




