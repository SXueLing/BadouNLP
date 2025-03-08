import torch
from transformers import BertModel

# 加载预训练的BERT模型
model = BertModel.from_pretrained('bert-base-chinese')

# 计算模型的总参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 打印总参数量
total_params = count_parameters(model)
print(f"Total number of trainable parameters in BERT-base model: {total_params}")
