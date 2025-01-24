# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "./张志龙/week7/output",
    "train_data_path": "./张志龙/week7/train.csv",
    "valid_data_path": "./张志龙/week7/test.csv",
    "vocab_path":"./张志龙/week7/chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 3,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": r"./张志龙/week7/bert-base-chinese",
    "seed": 987
}

