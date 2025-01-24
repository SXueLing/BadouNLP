from transformers import BertModel

model = BertModel.from_pretrained(r"d:\大数据八斗\bert-base-chinese", return_dict=False)
n = 2                       # 输入最大句子个数
vocab = 21128               # 词表数目
max_sequence_length = 512   # 最大句子长度
embedding_size = 768        # embedding维度
hide_size = 3072            # 隐藏层维数
num_layers = 12             # 隐藏层层数


# embedding层参数 (token层+position层+segment层) 再加一个归一化层
embedding_parameters = (vocab * embedding_size) + (max_sequence_length * embedding_size) + (n * embedding_size) + (embedding_size + embedding_size)
# self-attention层参数 (768*768+768)*3 也就是 q k v 各一层
self_attention_parameters = (embedding_size * embedding_size + embedding_size) * 3
# self-attention输出层参数 (过一个线性层 然后过一个归一化层)
self_attention_out_parameters = (embedding_size * embedding_size + embedding_size) + (embedding_size + embedding_size)
# feed-forward层参数 两个线性层+一个归一化层
feed_forward_parameters = (embedding_size * hide_size + hide_size) + (embedding_size * hide_size + embedding_size) + (embedding_size + embedding_size)
# pool层参数 (768*768+768)
pool_fc_parameters = (embedding_size * embedding_size + embedding_size)

all_paramerters = embedding_parameters + (self_attention_parameters + self_attention_out_parameters + \
    feed_forward_parameters) * num_layers + pool_fc_parameters

print("模型真实参数量%d" % sum(p.numel() for p in model.parameters()))
print("自己算的参数量%d" % all_paramerters)
