import torch
import torch.nn as nn
from transformers import BertModel

model = BertModel.from_pretrained(r"F:\Desktop\work_space\pretrain_models\bert-base-chinese", return_dict=False)

num_sentences = 2
vocab_size = 21128
max_seq_len = 512
embed_dim = 768
hidden_dim = 3072
num_hidden_layers = 12

embedding_params = vocab_size * embed_dim + max_seq_len * embed_dim + num_sentences * embed_dim + 2 * embed_dim
attention_params = (embed_dim * embed_dim + embed_dim) * 3
attention_output_params = embed_dim * embed_dim + 2 * embed_dim + 2 * embed_dim
feedforward_params = embed_dim * hidden_dim + hidden_dim + embed_dim * hidden_dim + 3 * embed_dim
pooling_params = embed_dim * embed_dim + embed_dim

calculated_params = embedding_params + (attention_params + attention_output_params + feedforward_params) * num_hidden_layers + pooling_params

actual_params = sum(p.numel() for p in model.parameters())
print(f"Actual model parameter count: {actual_params}")
print(f"Calculated parameter count: {calculated_params}")
