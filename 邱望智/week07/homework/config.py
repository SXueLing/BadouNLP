# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "data_path": "../文本分类练习数据集/文本分类练习.csv",
    "train_data_path": "../文本分类练习数据集/训练集.csv",
    "valid_data_path": "../文本分类练习数据集/测试集.csv",
    "output_path": "output/结果统计.csv",
    "vocab_path": "chars.txt",
    "model_type": "bert_lstm",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "pretrain_model_path": r"D:\bert-base-chinese",
    "seed": 987
}
