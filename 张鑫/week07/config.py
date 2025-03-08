"""
配置参数信息
"""

Config = {
    # 路径设置
    "model_path": "output",
    "original_data_path": "文本分类练习.csv",
    "train_data_path": "data/new_train_tag_news.json",
    "valid_data_path": "data/new_valid_tag_news.json",
    "vocab_path": "chars.txt",
    "pretrain_model_path": r"E:\nlp22\week06\bert-base-chinese",
    # label设置
    "index_to_label": {1: "好评", 0: "差评"},
    # 模型设置
    "model_type": "bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 128,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "seed": 987
}