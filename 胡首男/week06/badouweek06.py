from transformers import BertModel, BertConfig

# 配置BERT模型
config = BertConfig(
    hidden_size=768,       # 隐藏层大小
    num_attention_heads=12, # 注意力头数
    num_hidden_layers=12,   # 层数
    intermediate_size=3072, # 前馈网络的维度
    vocab_size=30522       # 词汇表大小
)

# 加载BERT模型
model = BertModel.from_pretrained(r"F:\BaiduNetdiskDownload\八斗精品课(1)\第六周 语言模型\bert-base-chinese", return_dict=False)

# 计算总参数数量
total_params = sum(p.numel() for p in model.parameters())

print(f"Total number of parameters: {total_params}")
