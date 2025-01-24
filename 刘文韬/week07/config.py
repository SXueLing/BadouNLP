# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": r"E:\日常\学习\八斗\homework\week_7\data\train_data.csv",
    "valid_data_path": r"E:\日常\学习\八斗\homework\week_7\data\test_data.csv",
    "vocab_path":"chars.txt",
    "model_type":"rcnn",
    "max_length": 100,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 5,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"E:\日常\学习\八斗\第六周 语言模型\bert-base-chinese\bert-base-chinese",
    "seed": 987
}

