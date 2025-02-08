# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "provide_csv_path": r"../文本分类练习数据集/文本分类练习.csv",
    "train_ratio": 0.8,
    "train_data_csv_path": r"../文本分类练习数据集/训练数据.csv",
    "evaluate_data_csv_path": r"../文本分类练习数据集/测试数据.csv",
    # "model_type": "gated_cnn",
    "model_type": "bert",
    "pretrain_model_path": r"F:\NLP\资料\week6 语言模型和预训练\bert-base-chinese",
    "max_length": 25,
    "model_path": "output",
    "class_num": 2,
    "num_layers": 2,
    "pooling_style": "max",
    "optimizer": "adam",
    "epoch": 10,
    "hidden_size": 256,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "vocab_size": 4622,
    "vocab_path": "../chars.txt",


    # "train_data_path": "../train_tag_news.json",
    # "valid_data_path": "../valid_tag_news.json",

    "kernel_size": 3,
    "seed": 1024
}
