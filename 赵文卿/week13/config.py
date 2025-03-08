# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "week13/output",
    "schema_path": "week9/data/schema.json",
    "train_data_path": "week9/data/train",
    "valid_data_path": "week9/data/test",
    "vocab_path":"week9/data/chars.txt",
    "model_type":"bert",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers" : 2,
    "epoch": 20,
    "batch_size": 64,
    "tuning_tactics":"lora_tuning",
    #"tuning_tactics":"prompt_tuning",
    #"tuning_tactics":"prefix_tuning",
    #"tuning_tactics":"p_tuning",
    # "tuning_tactics":"finetuing",
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "class_num": 9,
    "pretrain_model_path":r"E:\bert-base-chinese",
    "seed": 987
}

