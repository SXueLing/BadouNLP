# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "data_path": "data/reviews_data.csv",
    "train_data_path": "data/train_reviews.csv",
    "valid_data_path": "data/valid_reviews.csv",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 33,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"D:\NLP\pretrain_model\bert-base-chinese",
    "seed": 987
}

