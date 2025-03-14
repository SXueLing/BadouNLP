# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "schema_path": "ner_data/schema.json",
    "vocab_path": "chars.txt",
    "model_type": "bert",
    "max_length": 100,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 32,
    "tuning_tactics": "lora_tuning",
    # "tuning_tactics":"finetuing",
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": "/Users/duli/Desktop/UT Austin/自学内容/八斗课件/NLP/six-lm/bert-base-chinese",
    "seed": 987
}