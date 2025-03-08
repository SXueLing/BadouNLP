# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": r"E:/八斗课程/第七周/train_data.json",
    "valid_data_path": r"E:/八斗课程/第七周/test_data.json",
    "vocab_path":r"E:/八斗课程/第七周/nn_pipline/chars.txt",
    "model_type":"fast_text",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"F:/八斗课程/第七周/文本分类问题/bert-base-chinese",
    "seed": 987
}
