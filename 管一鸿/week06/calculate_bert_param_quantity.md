## 计算Bert中参数量的思路

​																																																	管一鸿  24.12.29

### 一、超参数定义

1. hidden_size  = 768
2. h = 12 # 注意力头数量 multi header counts
3. d_k = hidden_size / h = 64 #  多头机制中每个头的维度
4. L #文本序列长度
5. num_layers # transformer层数
6. inter_size = 4*hidden_size = 3072 # feed forword 中间层维度

### 二、分层计算参数量

1. #### embedding

   | name                 | SHAPE[0] | SHAPE[1]    | total                   |
   | -------------------- | -------- | ----------- | ----------------------- |
   | word embedding       | L        | hidden_size | L x hidden_size         |
   | position embedding   | L        | hidden_size | L x hidden_size         |
   | token type embedding | L        | hidden_size | L x hidden_size         |
   | **total**            |          |             | **3 x L x hidden_size** |

2. #### embedding层后的Layer normalization

   1. weight hidden_size
   2. bias hidden_size
   3. **total hidden_size x 2**

3. #### transfomer

   1. self-attention

      | name      | SHAPE[0]    | SHAPE[1]    | BIAS        | total                                             |
      | --------- | ----------- | ----------- | ----------- | ------------------------------------------------- |
      | Query     | hidden_size | hidden_size | hidden_size | hidden_size x hidden_size + hidden_size           |
      | Key       | hidden_size | hidden_size | hidden_size | hidden_size x hidden_size + hidden_size           |
      | Value     | hidden_size | hidden_size | hidden_size | hidden_size x hidden_size + hidden_size           |
      | **total** |             |             |             | **3 x (hidden_size x hidden_size + hidden_size)** |

   2. multi_head output 线性层

      1. weight hidden_size x hidden_size
      2. bias hidden_size
      3. **total hidden_size x hidden_size + hidden_size**

   3. self-attention后的残差层的Layer normalize

      1. weight hidden_size
      2. bias hidden_size
      3. **total hidden_size x 2**

   4. Feed forward中的两个线性层

      | name      | SHAPE[0]    | SHAPE[1]    | BIAS        | total                                                        |
      | --------- | ----------- | ----------- | ----------- | ------------------------------------------------------------ |
      | 中间层    | inter_size  | hidden_size | inter_size  | inter_size x hidden_size + inter_size                        |
      | 输出层    | hidden_size | inter_size  | hidden_size | hidden_size x inter_size + hidden_size                       |
      | **total** |             |             |             | **2 x (hidden_size x inter_size) + inter_size + hidden_size** |

   5. Feed forward后的残差层的Layer normalize

      1. weight hidden_size
      2. bias hidden_size
      3. **total hidden_size x 2**

### 三、总结

```
bert_param_quantity = 3 x L x hidden_size + 2 x hidden_size +
					num_layers x   
					[ 3 x (hidden_size x hidden_size + hidden_size) + 
					hidden_size x hidden_size + hidden_size + 2 x (hidden_size x inter_size) + 
					inter_size + hidden_size + 2 x hidden_size ]
				  = (3L + 11 x num_layer + 2) x hidden_size + 12 x num_layer x hidden_size^2
```

代入hidden_size = 768，L = 30000，num_layer = 1 计算得：bert_param_quantity = 76,207,872  大约为0.08B。

结论：bert-base的参数量只与三个变量有关，分别是hidden_size，词表大小，transformer层的数量，而注意力头的数量和前馈网络中间层的维度与参数大小无关，但可能会影响训练效果。
