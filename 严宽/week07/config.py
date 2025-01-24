# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "train_set.csv",
    "valid_data_path": "test_set.csv",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 40,
    "hidden_size": 768,
    "kernel_size": 3,
    "num_layers": 4,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "pretrain_model_path":r"D:\learning\AI\第六周 语言模型\bert-base-chinese",
    "seed": 987
}

