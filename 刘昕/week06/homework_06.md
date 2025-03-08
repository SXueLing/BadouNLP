# homework_06

## 分析每一层的参数个数

len = 4

hid_size = 768

inter_size = 4 × hid_size = 3072

### Embeddings

#### 三层 Embeddings

1. Token Embeddings: (len, hid_size)
2. Segment Embeddings: (len, hid_size)
3. Position Embeddings: (len, hid_size)

#### Embedding 归一化输出层

1. weights: (hid_size, hid_size)
2. bias: (hid_size)

### Transformer

#### Self Attention

1. q_w: (hid_size, hid_size)
2. q_b: (hid_size)
3. k_w: (hid_size, hid_size)
4. k_b: (hid_size)
5. v_w: (hid_size, hid_size)
6. v_b: (hid_size)
7. output_w: (hid_size, hid_size)
8. output_b: (hid_size)

#### Attention 归一化层

1. norm_w: (hid_size, hid_size)
2. norm_b: (hid_size)

#### 残差层

1. weights: (hid_size, hid_size)
2. bias: (hid_size)

#### 前反馈层

1. inter_w: (inter_size, hid_size)
2. inter_b: (inter_size)
3. output_w: (hid_size, inter_size)
4. output_b: (hid_size)

#### 前反馈归一化层

1. norm_w: (hid_size, hid_size)
2. norm_b: (hid_size)

### 池化层

1. pooler_w: (hid_size, hid_size)
2. pooler_b: (hid_size)

## 计算参数个数

1. Embeddings: 599808

2. Transformer: 8856576

3. 池化层: 590592

4. 由于Transformer进行了12轮，所以总的参数个数为：

   $599808 + 12 \times 8856576 + 590592 = 107469312$

   大概是107b

