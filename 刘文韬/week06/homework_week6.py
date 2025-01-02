# bert参数公式

# 词表大小
vl = 21128
# 最大句子长度
max_sequence_length = 512
# embedding维度
H = 768

total_param_nums = H*(vl+max_sequence_length+18) + 13*H*H

print(total_param_nums)
