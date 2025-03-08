# 获取bert总共的参数量

def getBertParamNum(num_layers, hidden_size, vocab_size=21128):
    x = 0
    # embedding
    # word embedding
    x += vocab_size * hidden_size 
    # position embedding
    x += 512 * hidden_size 
    # segment embedding
    x += 2 * hidden_size
    # layer norm
    x += hidden_size * 2 # gamma, beta

    # transformer
    # qkv
    t = hidden_size * hidden_size + hidden_size # w,b
    t = t * 3
    # attention out
    t += hidden_size * hidden_size + hidden_size # w,b
    # layer norm
    t += hidden_size * 2 # gamma, beta
    # feed forward
    t += hidden_size * hidden_size * 4 + 4 * hidden_size # w,b 
    t += hidden_size * 4 * hidden_size + hidden_size # w,b
    # layer norm
    t += hidden_size * 2 # gamma, beta
    # sequence
    x += t * num_layers
    # pooler 
    x += hidden_size * hidden_size + hidden_size # w,b
    return x

print(getBertParamNum(1, 768))
