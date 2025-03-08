from transformers import BertModel, BertConfig

config = BertConfig()
model = BertModel(config)
total_params = sum(p.numel() for p in model.parameters())
print(f"参数量: {total_params}")
