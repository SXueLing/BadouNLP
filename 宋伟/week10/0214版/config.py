from transformers import BertConfig

def get_config():
    # 加载BERT的配置，设置一些自定义参数
    config = BertConfig(
        vocab_size=30522,  # BERT的词汇表大小
        hidden_size=768,   # 隐藏层大小
        num_hidden_layers=12,  # Transformer层数
        num_attention_heads=12,  # 自注意力头数
        intermediate_size=3072,  # 前馈层大小
        hidden_dropout_prob=0.1,  # Dropout概率
        attention_probs_dropout_prob=0.1,  # Attention Dropout概率
        max_position_embeddings=512,  # 最大位置编码长度
        type_vocab_size=2,  # 类型词汇表大小（用于区分句子A和句子B）
        initializer_range=0.02,  # 初始化范围
    )
    return config
