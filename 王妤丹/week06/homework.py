"""
计算bert需要多少参数量
根据hidden_size和num_layer计算参数量
"""
import math
def cal_transform_param(hidden_size, num_layer):
    total_promal = 0
    layer_promal = 0  #每一层参数量
    ## K Q V
    q_w = hidden_size*hidden_size
    q_b = hidden_size
    k_w = hidden_size*hidden_size
    k_b = hidden_size
    v_w = hidden_size*hidden_size
    v_b = hidden_size
    attention_w = hidden_size*hidden_size
    attention_b = hidden_size
    attention_layer__normal_w = hidden_size
    attention_layer__normal_b = hidden_size
    feed_forward_w = hidden_size*hidden_size*4
    feed_forward_b = hidden_size*4
    output__w = hidden_size*hidden_size*4
    output_b = hidden_size
    feed_forward_layer__normal_w = hidden_size
    feed_forward_layer__normal_b = hidden_size
    layer_promal = q_w + q_b + k_w + k_b + v_w + v_b + attention_w + attention_b + attention_layer__normal_w + attention_layer__normal_b + feed_forward_w + feed_forward_b + output__w + output_b + feed_forward_layer__normal_w + feed_forward_layer__normal_b
    total_promal = layer_promal * num_layer
    return total_promal
def cal_embedding_param(vocab_size, hidden_size,max_len): #embedding参数量
    token_embedding_w = 2*hidden_size
    word_embedding_w = vocab_size*hidden_size
    position_embedding_w = max_len*hidden_size
    return token_embedding_w + word_embedding_w + position_embedding_w
def cal_total_param(vocab_size, hidden_size, num_layer, max_len):
    return cal_embedding_param(vocab_size, hidden_size, max_len) + cal_transform_param(hidden_size, num_layer)
if __name__ == '__main__':
    vocab_size = 21128
    hidden_size = 768
    max_len = 512
    num_layer = 12
    print("vocab_size:%d, hidden_size:%d, max_len:%d, num_layer:%d"%(vocab_size, hidden_size, max_len, num_layer))
    embedding_param = cal_embedding_param(vocab_size, hidden_size, max_len)
    transformer_param = cal_transform_param(hidden_size, num_layer)
    total_param = embedding_param + transformer_param
    print("embedding参数量：%dM" % (embedding_param/(1024*1024)))
    print("transform参数量：%dM" % (transformer_param/(1024*1024)))
    print("总参数量：%dM" % (total_param/(1024*1024)))
