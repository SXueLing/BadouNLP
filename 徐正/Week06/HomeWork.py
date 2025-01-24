
count_bert = 0
count_transformer = 0
#self-attention 部分
weight_dim = 768
count_self_attention = 3 * weight_dim*(weight_dim + 1)

liner_dim = 768
count_transformer += count_self_attention + liner_dim*liner_dim
#Feed_forward部分
count_feed_forward = 768*3072
count_transformer += count_feed_forward * 2#两层线性层从shape上来看是转置

#embeding部分
len_vocab = 100
count_para_wordEmbeding = len_vocab * 768
count_para_SegAndPosition = 768*(512 + 2)
lyer_num = 1
count_bert = lyer_num * count_transformer + count_para_wordEmbeding + count_para_SegAndPosition

print(count_bert)
