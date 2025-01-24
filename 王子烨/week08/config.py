# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "../data/schema.json",
    "train_data_path": "../data/train.json",
    "valid_data_path": r"D:\code\pycharm\deep_learning\week8\triplet_loss\test_samples.json",
    "vocab_path": "../chars.txt",
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 10,
    "batch_size": 2,
    "epoch_data_size": 200,  # 每轮训练中采样数量
    "positive_sample_rate": 0.5,  # 正样本比例
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "vocab_size": 4622
}
