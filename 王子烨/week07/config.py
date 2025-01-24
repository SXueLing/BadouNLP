# -*- coding: utf-8 -*-
# @Time    : 2025/1/8 16:10
# @Author  : yeye
# @File    : config.py
# @Software: PyCharm
# @Desc    :
# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": r"D:\code\pycharm\NLP\week7\test\train_data.csv",
    "valid_data_path": r"D:\code\pycharm\NLP\week7\test\valid_data.csv",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 768,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"E:\model\bert-base-chinese",
    "seed": 987
}

