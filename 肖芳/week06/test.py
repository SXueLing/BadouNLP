from transformers import BertModel

vector_size = 768   # 向量维度
vocab_size = 21128  # 词表大小
layer_num = 12  # 层数
token_weight = vocab_size * vector_size
segment_weight = 2 * vector_size
position_weight = 512 * vector_size

# token_weight，segment_weight，position_weight 加和后过LN，输出维度为L*768
LN_embedding_weight = 2 * vector_size


### 过embedding层之后参数总量
embedding_weight = token_weight + segment_weight + position_weight + LN_embedding_weight

# 过attention(Q,K,V及多头机制)层，输出维度为L*768
weight_q = vector_size * vector_size + vector_size
weight_k = vector_size * vector_size + vector_size
weight_v = vector_size * vector_size + vector_size


# 过一个Linear层，输出维度为L*768
linear_after_attention = vector_size * vector_size + vector_size


# (残差机制)将过attention层的输出与输入相加，并且过LN，输出维度为L*768
LN_attention_weight = 2 * vector_size


### 过embedding层之后参数总量
sum_attention_weight = (weight_q + weight_k + weight_v + linear_after_attention + LN_attention_weight)

# 过FeedForward层(w=768*3072)，输出维度为L*3072
feed_forward_weight = vector_size * (vector_size * 4) + (vector_size * 4) 

# 过gelu激活函数，输出维度为L*3072
# 过Linear层(W=3072*768)，输出维度为L*768
linear_after_feed_forward = (vector_size*4) * vector_size + vector_size

# (残差机制)将过FeedForward层的输出与输入相加，并且过LN，输出维度为L*768
LN_feed_forward_weight = 2 * vector_size


### 过FeedForward层之后参数总量
sum_feed_forward_weight = (feed_forward_weight + linear_after_feed_forward + LN_feed_forward_weight)

# 单层transformer的参数总量
single_layer_weight = sum_attention_weight + sum_feed_forward_weight

# 过final_linear层，输出维度为L*768
final_linear_weight = vector_size * vector_size + vector_size

total_params = embedding_weight + single_layer_weight * layer_num + final_linear_weight

print("total_params", total_params)
print(f"这是一个{total_params/1024/1024}MB的模型")

bert = BertModel.from_pretrained(r"../bert-base-chinese", return_dict=False)
bert_params = sum(p.numel() for p in bert.parameters())
print(f"这是一个{bert_params/1024/1024}MB的模型")