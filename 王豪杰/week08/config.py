# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "week08/model_output",
    "schema_path": "./data/schema.json",
    "train_data_path": "./data/train.json",
    "valid_data_path": "./data/valid.json",
    "vocab_path":"./chars.txt",
    "max_length": 20,
    "hidden_size": 768,
    "epoch": 50,
    "batch_size": 64,
    "epoch_data_size": 400,     #每轮训练中采样数量
    "positive_sample_rate":0.5,  #正样本比例
    "optimizer": "adam",# sgd  adam
    "learning_rate": 1e-3,
}
