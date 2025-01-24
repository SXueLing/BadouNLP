
def bert_arg(v, d, L, df, max_embding):
    """
    v: 词表大小
    d: 词向量维度
    L: 文本长度Tranformer的层数
    df: 神经网络维度
    max_embding: 最大位置数
    """
    # 词嵌入维度
    word = v * d
    # 位置嵌入维度
    position = max_embding * d  
    #段嵌入维度
    segment = 2 * d
    # 注意力维度
    attention = 4 * d * d  
    # 前馈维度
    feed_forward = d*df+df*d
    # 层归一化
    layer_norm = 2 * d

    # 输出层
    output = d * v

    # 总参数
    total = word + position + segment + L * (attention + feed_forward + layer_norm)+ output
    return total

print(bert_arg(30522, 768, 12, 3072, 512))
