# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "data_path": "data/文本分类练习.csv",
    "train_data_percentage": 0.9,
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 128,
    "pooling_style":"avg",
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "pretrain_model_path":r"/Volumes/komorebi/model/bert-base-chinese",
    "result_path":"output/comparison_result.txt",
    "seed": 987
}

