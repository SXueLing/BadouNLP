"""
param sen_len:文本长度
param hidden_size:Embedding隐藏层的维度
param vocab_len:词表长度
param layer_num:transformer 层数
"""
 all_param = layer_num * （vocab_len * hidden_size + 2 * hidden_size + 512 * hidden_size + 3 * hidden_size * hidden_size(Q、K、V 计算的参数) + hidden_size * hidden_size (Attention层需要过一个线性层) + hidden_size * 4 * hidden_size * 2 (feed forward层的两个线性层)）
