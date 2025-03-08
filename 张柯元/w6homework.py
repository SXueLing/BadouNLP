## 计算 BERT中 有多少参数

def count_bert_params(vocab_size, hidden_size, num_layers, max_position_embeddings):

    # embedding层参数
     # token embedding层参数
    token_params = vocab_size * hidden_size

    # segment embedding层参数
    segment_params = 2 * hidden_size

    # position embedding层参数
    position_params = max_position_embeddings * hidden_size   ##(max_position_embedding = 512)
    embedding_params = token_params + segment_params + position_params


    # Transformer层参数

    q_params = hidden_size * hidden_size + 1 * hidden_size
    k_params = hidden_size * hidden_size + 1 * hidden_size
    v_params = hidden_size * hidden_size + 1 * hidden_size
    transformer_params = q_params + k_params + v_params

    # feedforward层参数
    feedforward_params = hidden_size * hidden_size *4 + 1 * hidden_size * 4
    feedforward_params += hidden_size*4 * hidden_size + 1 * hidden_size


    # 总参数数量
    total_params = embedding_params+ transformer_params * num_layers + feedforward_params
    return total_params



# BERT配置
vocab_size = 30522
hidden_size = 768
num_layers = 12
max_position_embeddings = 512

# 计算参数数量
total_params = count_bert_params(vocab_size, hidden_size, num_layers, max_position_embeddings)
print(f"Total number of parameters in BERT-Base: {total_params}")
