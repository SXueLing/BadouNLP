# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "E:/nlp_learn/practice/week07/nn_pipline/data/train_textdata.json",
    "valid_data_path": "E:/nlp_learn/practice/week07/nn_pipline/data/val_textdata.json",
    "vocab_path":"E:/nlp_learn/practice/week07/nn_pipline/chars.txt",
    "model_type":"bert",
    "max_length": 465,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "pretrain_model_path":r"E:\nlp_learn\practice\week07\bert-base-chinese",
    "seed": 987
}

