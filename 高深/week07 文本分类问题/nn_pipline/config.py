# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "../data/train_文本分类练习.json",
    "valid_data_path": "../data/valid_文本分类练习.json",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 5,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-5,
    # "pretrain_model_path":r"F:\Desktop\work_space\pretrain_models\bert-base-chinese",
    "pretrain_model_path":"bert-base-chinese",
    "seed": 987
}

