"""
@Project ：cgNLPproject
@File    ：week06_03.py
@Date    ：2025/01/02 10:15
nnlm by myself
bert的
"""
import numpy

def bert_embedding(batch_size, sentence_length, vocab_length, max_length_,hidden_size):
    # x.shape=sentence_length 过word_embedding、position_embedding、token_embedding层，得：word_embedding_out、position_embedding_out、token_embedding_out
    word_embedding_weight = vocab_length * hidden_size
    word_embedding_bias = vocab_length
    position_embedding_weight = max_length_ * hidden_size
    position_embedding_bias = max_length_
    token_embedding_weight = 2 * hidden_size
    token_embedding_bias = 2
    word_embedding_out = batch_size * sentence_length * hidden_size
    position_embedding_out = batch_size * sentence_length * hidden_size
    token_embedding_out = batch_size * sentence_length * hidden_size
    # 三个embedding_out相加后，过归一化层
    embedding_layer_norm_weight = hidden_size * hidden_size
    embedding_layer_norm_bias = hidden_size
    embedding_layer_norm_out = batch_size * sentence_length * hidden_size
    return sum([word_embedding_weight,word_embedding_bias,position_embedding_weight,position_embedding_bias,
                token_embedding_weight,token_embedding_bias,word_embedding_out,position_embedding_out,
               token_embedding_out,embedding_layer_norm_weight,embedding_layer_norm_bias,embedding_layer_norm_out])


def bert_transform(batch_size, sentence_length, hidden_size,intermediate_size, num_attention_head):
    # x.shape = sentence_length * hidden_size
    # x拆分成12块后，每块.shape = num_attention_head,sentence_length,attention_head_size，
    # 分别过Q、K、V三层，得：q_out、k_out、v_out，再拼接成sentence_length * hidden_size
    q_weight = hidden_size * hidden_size
    q_bias = hidden_size
    k_weight = hidden_size * hidden_size
    k_bias = hidden_size
    v_weight = hidden_size * hidden_size
    v_bias = hidden_size
    attention_head_size = hidden_size / num_attention_head
    q = batch_size * num_attention_head * sentence_length * attention_head_size
    k = batch_size * num_attention_head * sentence_length * attention_head_size
    v = batch_size * num_attention_head * sentence_length * attention_head_size
    # dk = hidden_size/num_attention_head 是个常数
    dk = 1
    # qkv=softmax((q*k.T)/math.sqrt(dk)) * v
    qkv = batch_size * sentence_length * hidden_size
    # x与qkv相加后，过attention_layer_norm归一化层，得attention_layer_norm_out
    attention_layer_norm_weight = hidden_size * hidden_size
    attention_layer_norm_bias = hidden_size
    attention_layer_norm_out = batch_size * sentence_length * hidden_size
    # 进入feed forward层 实际上是一个全连接层intermediate_layer，一个激活函数(此激活函数在初始化时选定)，得intermediate_out。
    # 然后再连接一个全连接层output_layer，过dropout后，得feedforward_out
    intermediate_weight = hidden_size * intermediate_size
    intermediate_bias = intermediate_size
    # intermediate_out.shape = sentence_size * intermediate_size，过激活函数，.shape = sentence_length * intermediate_size
    intermediate_out = batch_size * sentence_length * intermediate_size
    output_weight = intermediate_size * hidden_size
    output_bias = hidden_size
    output_out = batch_size * sentence_length * hidden_size
    # attention_layer_norm_out与output_out相加，过feedforward归一化层
    feedforward_layer_norm_weight = hidden_size * hidden_size
    feedforward_layer_norm_bias = hidden_size
    feedforward_layer_norm_out = batch_size * sentence_length * hidden_size
    return sum([q_weight,q_bias,k_weight,k_bias,v_weight,v_bias,q,k,v,qkv,dk,
                attention_layer_norm_weight,attention_layer_norm_bias,attention_layer_norm_out,
                intermediate_weight,intermediate_bias,intermediate_out,
                output_weight,output_bias,output_out,
                feedforward_layer_norm_weight,feedforward_layer_norm_bias,feedforward_layer_norm_out])

def bert_pooler(batch_size, hidden_size):
    # feedforward_layer_norm_out.shape = 1, hidden_size
    pooler_layer_weight = hidden_size * hidden_size
    pooler_layer_bias = hidden_size
    pooler_layer_out = batch_size * 1 * hidden_size
    # 再过tanh函数 还是1 * hidden_size
    return sum([pooler_layer_weight, pooler_layer_bias, pooler_layer_out])


def main(batch_size_, sentence_length_, vocab_length_, max_length_, hidden_size_, intermediate_size_, num_attention_head_, transformer_layer_num_):
    embedding_count = bert_embedding(batch_size_, sentence_length_, vocab_length_, max_length_, hidden_size_)
    transformer_count = bert_transform(batch_size_, sentence_length_, hidden_size_, intermediate_size_, num_attention_head_)
    pooler_count = bert_pooler(batch_size_, hidden_size_)
    total = embedding_count + transformer_layer_num_ * transformer_count + pooler_count
    print('total:',total)

if __name__ == '__main__':
    # x = [[1,2,3,4],[1,2,3,4]] # shape: (2,4)
    # x = [[1,2,3,4]] # shape: (1,4)
    x = numpy.array([1, 2, 3, 4])
    transformer_layer_num = 1
    hidden_size = 768
    max_length = 512
    vocab_length = 21128
    num_attention_head = 12
    intermediate_size = 3072  # 4*768
    batch_size = 20
    sentence_length = x.size
    attention_head_size = hidden_size / num_attention_head
    main(batch_size, sentence_length, vocab_length, max_length, hidden_size, intermediate_size, num_attention_head, transformer_layer_num)

    embedding_weight_bias_count = vocab_length * hidden_size + vocab_length + max_length * hidden_size + max_length + 2 * hidden_size + 2
    embedding_out = 4 * batch_size * sentence_length * hidden_size
    embedding_layer_norm_weight_bias = hidden_size * hidden_size + hidden_size
    embedding_layer_norm_out = embedding_weight_bias_count + embedding_out + embedding_layer_norm_weight_bias

    transformer_q_k_v_layer_norm_weight_bias_count = 5 * hidden_size * hidden_size + 5 * hidden_size
    transformer_2_linear_weight_bias_count = 2 * intermediate_size * hidden_size + intermediate_size + hidden_size
    transformer_intermediate_linear_out_count = batch_size * sentence_length * intermediate_size
    transformer_q_k_v = 3 * batch_size * num_attention_head * sentence_length * attention_head_size
    transformer_out = 4 * batch_size * sentence_length * hidden_size
    dk = hidden_size / num_attention_head

    transformer_count = (transformer_q_k_v_layer_norm_weight_bias_count + transformer_2_linear_weight_bias_count + transformer_intermediate_linear_out_count
                     + transformer_q_k_v + transformer_out + dk)

    pooler_count = hidden_size * hidden_size + hidden_size + batch_size * hidden_size
    total_2 = embedding_layer_norm_out + transformer_count + pooler_count
    print('total_2:', total_2)
    q_k_v = batch_size * num_attention_head * sentence_length * attention_head_size
    transformer_out_without_qkv = 4 * batch_size * sentence_length * hidden_size

