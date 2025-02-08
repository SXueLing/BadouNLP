# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "data/train.xlsx",
    "valid_data_path": "data/evaluate.xlsx",
    "predict_data_path": "data/predict.xlsx",
    "vocab_path":"chars.txt",
    "model_type":"bert_cnn",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 3,
    "batch_size": 64,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "pretrain_model_path":"bert-base-chinese",
    "seed": 987
}

