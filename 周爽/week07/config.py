# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": r"D:\000nlpStudy\week7文本分类问题\nn_pipline\output",
    "train_data_path": r"D:\000nlpStudy\week7文本分类问题\data\train_data.json",
    "valid_data_path": r"D:\000nlpStudy\week7文本分类问题\data\valid_data.json",
    "vocab_path":r"D:\000nlpStudy\week7文本分类问题\nn_pipline\chars.txt",
    "model_type":"rcnn",
    "max_length": 30,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 1,
    "epoch": 15,
    "batch_size": 100,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"D:\bert-base-chinese",
    "seed": 987
}

