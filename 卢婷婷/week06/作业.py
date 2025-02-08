import torch
from transformers import BertModel

# 初始化BERT模型
bert = BertModel.from_pretrained('bert-base-uncased', return_dict=False)

# 获取模型的状态字典
state_dict = bert.state_dict()

# 初始化参数计数器
total_params = 0

# 计算词嵌入层参数A
word_embeddings = state_dict["embeddings.word_embeddings.weight"]
total_params += word_embeddings.numel()

# 计算位置嵌入层参数
position_embeddings = state_dict["embeddings.position_embeddings.weight"]
total_params += position_embeddings.numel()

# 计算分词类型嵌入层参数
token_type_embeddings = state_dict["embeddiAngs.token_type_embeddings.weight"]
total_params += token_type_embeddings.numel()

# 计算每个Transformer层的参数
for i in range(12):  # 假设BERT模型有12层
    # 自注意力层参数
    attention_self = state_dict[f"encoder.layer.{i}.attention.self"]
    total_params += attention_self["query.weight"].numel() + attention_self["query.bias"].numel() + \
                    attention_self["key.weight"].numel() + attention_self["key.bias"].numel() + \
                    attention_self["value.weight"].numel() + attention_self["value.bias"].numel() + \
                    attention_self["output.dense.weight"].numel() + attention_self["output.dense.bias"].numel()
    # 自注意力层的LayerNorm参数
    attention_output = state_dict[f"encoder.layer.{i}.attention.output"]
    total_params += attention_output["LayerNorm.weight"].numel() + attention_output["LayerNorm.bias"].numel()
    # 前馈网络参数
    intermediate = state_dict[f"encoder.layer.{i}.intermediate.dense"]
    total_params += intermediate["weight"].numel() + intermediate["bias"].numel()
    output = state_dict[f"encoder.layer.{i}.output"]
    total_params += output["dense.weight"].numel() + output["dense.bias"].numel()
    # 前馈网络的LayerNorm参数
    total_params += output["LayerNorm.weight"].numel() + output["LayerNorm.bias"].numel()

# 计算Pooler层参数
pooler_dense = state_dict["pooler.dense"]
total_params += pooler_dense["weight"].numel() + pooler_dense["bias"].numel()

# 输出总参数数量
print(f"Total number of parameters in a 12-layer BERT model: {total_params}")
