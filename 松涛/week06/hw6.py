# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-chinese")

model_parameters = 0
for p in model.parameters():
    model_parameters += p.numel()
    
print(f"Official # of parameters: {model_parameters}")

vocab_size = 21128
max_input_len = 512
embedding_size = 768
hidden_size = 3072
num_hidden_layers = 12

token_embedding = vocab_size * embedding_size
segment_embedding = 2 * embedding_size
position_embedding = max_input_len * embedding_size

layer_norm = embedding_size + embedding_size
# embedding过程总参数统计
embedding_parameters = token_embedding + segment_embedding + position_embedding + layer_norm

# self_attention过程的参数统计
self_attention_parameters = (embedding_size * embedding_size + embedding_size) * 3

# self_attention_out参数统计
self_attention_out_parameters = embedding_size * embedding_size + embedding_size + layer_norm

# Feed Forward参数统计
feed_forward_parameters = embedding_size * hidden_size + hidden_size + embedding_size * hidden_size + embedding_size + layer_norm

# pool_fc层参数统计
pool_fc_parameters = embedding_size * embedding_size + embedding_size

# 总参数统计
all_parameters = embedding_parameters + (self_attention_parameters + self_attention_out_parameters + feed_forward_parameters) * num_hidden_layers + pool_fc_parameters

print(f"Calculated # of parameters: {all_parameters}")