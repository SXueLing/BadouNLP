# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": r"C:\py_xm\nlp01\week7\文本分类练习.csv",
    "valid_data_path": "../data/valid_tag_news.json",
    "vocab_path":"chars.txt",
    "model_type":"bert_cnn",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 64,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"C:\Users\Administrator\Desktop\人工智能\week6 语言模型和预训练\bert-base-chinese",
    "seed": 987
}

