# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": r"H:\pyProj1\nlp_learn\week07\data\train_tag_homework.json",
    "valid_data_path": r"H:\pyProj1\nlp_learn\week07\data\valid_tag_homework.json",
    "vocab_path":r"H:\pyProj1\nlp_learn\week07\nn_pipline\chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",                       
    "learning_rate": 1e-3,
    "pretrain_model_path":r"H:\pyProj1\nlp_learn\week06\pm\pretrain_models",
    "seed": 987
}

