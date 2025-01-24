"""
配置信息
"""

Config = {
    #     模型目录
    'model_path': 'output',
    #     训练数据目录
    'train_data_path': 'data/train_data.json',
    #     验证数据目录
    'valid_data_path': 'data/test_data.json',
    #     词表目录
    'vocab_path': 'data/chars.txt',
    #     模型类型
    'model_type': 'bert',
    #     文本最大长度
    'max_length': 35,
    # 隐藏层维数
    "hidden_size": 768,
    "kernel_size": 3,
    "num_layers": 2,
    # 训练轮数
    "epoch": 15,
    # 批次大小
    "batch_size": 1024,
    # 池化类型, max or mean
    "pooling_style": "max",
    # 优化器
    "optimizer": "adam",
    "learning_rate": 0.00008,
    "pretrain_model_path": r"/mnt/d/myself/study/AI/model/bert-base-chinese",
    "seed": 256,

}
